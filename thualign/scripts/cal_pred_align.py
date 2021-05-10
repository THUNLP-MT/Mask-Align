import argparse
import thualign.utils as utils
from thualign.utils import alignment

def parse_args():
    parser = argparse.ArgumentParser(description="calculate acc and aer")

    parser.add_argument("pred_hyp", help="prediction hypothesis")
    parser.add_argument("pred_ref", help="prediction reference")
    parser.add_argument("align_hyp", help="alignment hypothesis")
    parser.add_argument("align_ref", help="alignment reference")

    return parser.parse_args()

def main(args):
    refs, poss = utils.parse_refs(args.align_ref)
    hyps, _ = utils.parse_refs(args.align_hyp)

    correct = []

    with open(args.pred_ref) as f_ref, open(args.pred_hyp) as f_hyp:
        for single_ref, single_hyp in zip(f_ref, f_hyp):
            single_ref = single_ref.strip().split()
            single_hyp = single_hyp.strip().split()
            ref_t, ref_group, ref_group_r = alignment.merge(single_ref)
            correct_t = [single_ref[i] == single_hyp[i] for i in range(len(single_ref))]
            correct_tt = []
            for group in ref_group_r:
                cur = False
                for g in group:
                    if correct_t[g]:
                        cur = True
                correct_tt.append(cur)
            correct.append(correct_tt)
    
    align_correct_t = [list(h & p) for h, p in zip(hyps, poss)]
    align_correct = []
    for align_correct_tt in align_correct_t:
        align_correct_c = []
        for x, y in align_correct_tt:
            align_correct_c.append(y-1)
        align_correct_c = list(set(align_correct_c))
        align_correct.append(align_correct_c)
    
    correct_align = [0, 0] # correct/wrong pred
    wrong_align = [0, 0] # correct/wrong pred
    for align, pred in zip(align_correct, correct):
        false_align = list(set(range(len(pred))) - set(align))
        for i in align:
            if pred[i]:
                correct_align[0] += 1
            else:
                correct_align[1] += 1
        for i in false_align:
            if pred[i]:
                wrong_align[0] += 1
            else:
                wrong_align[1] += 1
    print(correct_align)
    print(wrong_align)

if __name__ == "__main__":
    main(parse_args())