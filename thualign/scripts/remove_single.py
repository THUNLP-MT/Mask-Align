import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Remove one token sentence pair")

    parser.add_argument("src", help="input src")
    parser.add_argument("tgt", help="input tgt")
    parser.add_argument("--talp", help="input talp")
    parser.add_argument("--src_out", help="output src")
    parser.add_argument("--tgt_out", help="output tgt")

    return parser.parse_args()

def main(args):
    cnt = 0
    idx = 0
    src_out = args.src_out or args.src + '.clean'
    tgt_out = args.tgt_out or args.tgt + '.clean'
    if args.talp is None:
        with open(args.src) as sin, open(args.tgt) as tin, open(src_out, 'w') as sout, open(tgt_out, 'w') as tout:
            for s, t in zip(sin, tin):
                idx += 1
                if len(s.strip().split()) <= 1 or len(t.strip().split()) <= 1:
                    print("Find one token sent pair({}): <{}, {}>".format(idx, s, t))
                    cnt += 1
                else:
                    sout.write(s)
                    tout.write(t)
    else:
        with open(args.src) as sin, open(args.tgt) as tin, open(args.talp) as ain, open(src_out, 'w') as sout, open(tgt_out, 'w') as tout, open(args.talp + '.clean', 'w') as aout:
            for s, t, a in zip(sin, tin, ain):
                idx += 1
                if len(s.strip().split()) <= 1 or len(t.strip().split()) <= 1:
                    print("Find one token sent pair({}): <{}, {}>".format(idx, s, t))
                    cnt += 1
                else:
                    sout.write(s)
                    tout.write(t)
                    aout.write(a)

    print("Find one token sent pairs: {}".format(cnt))

if __name__ == "__main__":
    main(parse_args())
