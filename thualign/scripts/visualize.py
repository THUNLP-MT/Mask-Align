#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

import os
import time
import argparse
import remi.gui as gui
from remi import start, App

import torch
from collections.abc import Iterable
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.commons.utils import JsCode

def sort_keys(keys):
    """
    Specially designed for alignment datas
    """
    ref = []
    bi_align = []
    hard = []
    weight = []
    for k in keys:
        if 'ref' in k:
            ref.append(k)
        elif 'bi_align' in k:
            bi_align.append(k)
        elif '.hard' in k:
            hard.append(k)
        else:
            weight.append(k)
    ref = sorted(ref)
    bi_align = sorted(bi_align)
    hard = sorted(hard)
    weight = sorted(weight)

    # the latter, the higher level
    res = ref + weight + hard + bi_align
    return res

def legalize(s):
    return s.replace('<', '[').replace('>', ']').replace('▁','')

def legalize_lb(lb):
    '''
    sentence_piece style -> subword-nmt style
    example:
        ▁a b ▁c -> a @@b c
    '''
    if not lb[0].startswith('▁'):
        return lb
    res = []
    for i in lb:
        if i.startswith('▁'):
            res.append(i.replace('▁',''))
        else:
            res[-1] = res[-1] + '@@'
            res.append(i)
    return res

class DataGenerator:

    def __init__(self, data):
        self._data = data # ['src', 'tgt', 'weight', 'metrics', 'ref']
        for i in range(len(self._data)):
            for k in self._data[i]['metrics']:
                if not isinstance(self._data[i]['metrics'][k], Iterable):
                    self._data[i]['metrics'][k] = (self._data[i]['metrics'][k], )
        self.idx = -1

    def __len__(self):
        return len(self._data)

    def last(self):
        self.idx -= 1
        return self.__getitem__(self.idx)

    def next(self):
        self.idx += 1
        return self.__getitem__(self.idx)

    def __getitem__(self, x):
        while x < 0:
            x += len(self._data)
        if x >= len(self._data):
            x = x % len(self._data)
        self.idx = x
        data = self._data[x] # data: dict of weights [ny, nx]
        x_lb = legalize_lb(data['src'])
        y_lb = legalize_lb(data['tgt'])
        info = 'Info: '
        if 'metrics' in data:
            metrics = data['metrics']
            keys = sort_keys(metrics.keys())
            for k in keys:
                info += '{}: {:.1f} '.format(k, metrics[k][0]*100)      
        weights = {}
        for k, v in data['weights'].items():
            if isinstance(v, list):
                # [(j, i, v)]
                tw = v
            else:
                tw = v
                tw = tw[:len(y_lb), :len(x_lb)]
                tw = [(j, i, tw[i,j].item()) for i in range(tw.shape[0]) for j in range(tw.shape[1])]
            weights[k] = [{'name': '[{}, {}]'.format(legalize(x_lb[x[0]]), legalize(y_lb[x[1]])), 'value': [x[0], x[1], x[2]*100]} for x in tw]
        value = {
            'info': info,
            'weights': weights
        }
        return x_lb, y_lb, value

    def sorted_by(self, expr):
        try:
            scores = []
            for data in self._data:
                metrics = {}
                for k, v in data['metrics'].items():
                    metrics[k] = v[0]
                scores.append(eval(expr, metrics))
            self._data = [x for _, x in sorted(zip(scores, self._data), key=lambda pair: pair[0], reverse=True)]
            return True
        except Exception:
            return False

class AlignmentChart(gui.Container):

    def __init__(self, vizdata, chart_args={'width':2000, 'height':800, 'margin': '10px'}, *args, **kwargs):
        super(AlignmentChart, self).__init__(*args, **kwargs)
        self.chart_args = chart_args
        self.chart = self.get_chart()
        self.current_filename = str(time.time()) + '.html'
        all_data = torch.load(vizdata, map_location=torch.device('cpu'))
        all_data = sorted(all_data, key=lambda x: len(x['src']) + len(x['tgt']))
        self._data = DataGenerator(all_data)

    def get_index(self):
        return self._data.idx

    def get_chart(self):
        chart = gui.Widget( _type='iframe', **self.chart_args)
        chart.attributes['width'] = '100%'
        chart.attributes['height'] = '100%'
        chart.attributes['controls'] = 'true'
        chart.style['border'] = 'none'
        return chart

    def render(self, x_lb, y_lb, value):
        info = value['info'] if 'info' in value else ""
        c = (
            HeatMap(init_opts=opts.InitOpts(width='{}px'.format(max(50*len(x_lb), 1000)), height='{}px'.format(max(20*len(y_lb), 500))))
            .add_xaxis(x_lb)
            .set_global_opts(
                legend_opts=opts.LegendOpts(type_='scroll'),
                visualmap_opts=opts.VisualMapOpts(is_show=True, 
                                                orient='horizontal', 
                                                pos_left='center', 
                                                pos_bottom='-20',
                                                range_color=['#ffffff', '#000000']),
                tooltip_opts=opts.TooltipOpts(is_show=True, formatter=JsCode("""function(params){
                            return params.data['name'] + ' : ' + (params.data['value'][2] / 100).toFixed(2) 
                        }
                        """)),
                xaxis_opts=opts.AxisOpts(axislabel_opts={'rotate':45, 'interval': 0}, interval=0, splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts={'color':'black', 'width':1})),
                yaxis_opts=opts.AxisOpts(axislabel_opts={'interval': 0}, is_inverse=True, interval=0,
                                        splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts={'color':'black', 'width':1}))
            )
        )
        keys = sort_keys(value['weights'].keys())
        for k in keys:
            v = value['weights'][k]
            if 'ref' in k:
                v = [{'name': x['name'], 'value': [x['value'][0], x['value'][1], 0]} for x in v]
                itemOpts = opts.ItemStyleOpts(color="rgba(0,0,0,0)", border_color="black", border_width=4)
            else:
                itemOpts = opts.ItemStyleOpts(opacity=1.0, border_color="black", border_width=1)
            # if '.hard' in k:
            #     v = [{'name': x['name'], 'value': [x['value'][0], x['value'][1], 60]} for x in v]
            c.add_yaxis(
                k,
                yaxis_data=y_lb,
                value=v,
                is_selected=('ref' in k),
                itemstyle_opts=itemOpts
            )
        
        for filename in os.listdir(res_path):
            if os.path.splitext(filename)[-1] == '.html':
                os.remove(os.path.join(res_path, filename))
        self.current_filename = str(time.time()) + '.html'
        c.render(os.path.join(res_path, self.current_filename))
        self.remove_child('chart')
        self.chart = self.get_chart()
        self.chart.attributes['src'] = f'/{load_path}:' + self.current_filename
        self.append(self.chart, "chart")
        return info

    def update(self, idx=None, forward=True):
        if idx is not None:
            return self.render(*self._data[idx])
        if forward:
            return self.render(*self._data.next())
        else:
            return self.render(*self._data.last())

    def reorder(self, expr):
        success = self._data.sorted_by(expr)
        if success:
            return self.update(idx=0)
        else:
            return None

class MyApp(App):

    def __init__(self, *args):
        super(MyApp, self).__init__(*args, static_file_path={load_path:res_path})

    def main(self):
        self.verticalContainer = gui.Container(width=2000, margin='0px auto', style={'display': 'block', 'overflow': 'hidden'})

        # chart container
        self.chartContainer = gui.Container(width=2000, margin='0px auto', style={'display': 'block', 'overflow': 'hidden'})

        self.infoContainer = gui.Container(width='100%', layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='0px', style={'display': 'block', 'overflow': 'auto'})

        self.index_lb = gui.Label('Index: ', width=30, height=30, margin='10px')
        self.index_input = gui.Input(input_type='number', default_value=0, width=40, height=15, margin='10px')
        self.index_input.onchange.do(self.on_index_change)
        self.expr = gui.TextInput(width=500, height=15, margin='10px')
        self.expr.set_text('Type in a expression')
        self.expr.onchange.do(self.on_expr_change)

        self.info_lb = gui.Label('Info: ', width=2000, height=30, margin='10px')

        self.infoContainer.append([self.index_lb, self.index_input, self.expr])
        
        self.chart = AlignmentChart(vizdata, width=2000, height=1000, margin='10px')
        self.chartContainer.append(self.infoContainer)
        self.chartContainer.append(self.info_lb)
        self.chartContainer.append(self.chart, "chart")

        self.next_bt = gui.Button('Next', width=200, height=30, margin='10px')
        self.next_bt.onclick.do(self.on_next_button_pressed)
        self.last_bt = gui.Button('Last', width=200, height=30, margin='10px')
        self.last_bt.onclick.do(self.on_last_button_pressed)

        self.verticalContainer.append(self.chartContainer)
        self.verticalContainer.append(self.next_bt)
        self.verticalContainer.append(self.last_bt)
        return self.verticalContainer

    def on_index_change(self, widget, value):
        self.idx = int(self.index_input.get_value())
        info = self.chart.update(idx=self.idx)
        self.info_lb.set_text(info)
    
    def on_expr_change(self, widget, value):
        info = self.chart.reorder(value)
        if info:
            self.info_lb.set_text(info)

    def on_next_button_pressed(self, widget):
        self.idx = self.chart.get_index()
        self.idx += 1
        info = self.chart.update(idx=self.idx)
        self.index_input.set_value(self.idx)
        self.info_lb.set_text(info)

    def on_last_button_pressed(self, widget):
        self.idx = self.chart.get_index()
        self.idx -= 1
        info = self.chart.update(idx=self.idx)
        self.index_input.set_value(self.idx)
        self.info_lb.set_text(info)

    def data_select_dialog_confirm(self, widget):
        src, tgt, data = self.data_select_dialog.get_filenames()
        self.src_lb.set_text(src)
        self.tgt_lb.set_text(tgt)
        self.data_lb.set_text(data)
        self.chart.load_binary_data(*self.data_select_dialog.get_values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention visualization",
        usage="main.py [<args>] [-h | --help]"
    )

    # configure file
    parser.add_argument("vizdata", type=str)
    parser.add_argument("--port", type=int, default=8082)

    return parser.parse_args()

vizdata = None
res_path = os.getcwd()
load_path = 'data'

if __name__ == "__main__":
    args = parse_args()
    vizdata = args.vizdata
    start(MyApp, debug=True, address='0.0.0.0', port=args.port, start_browser=True, multiple_instance=True)