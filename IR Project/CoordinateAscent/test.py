from __future__ import print_function, division

import argparse

from sklearn.datasets import load_svmlight_file

from coordinate_ascent import CoordinateAscent
from metrics import MAPScorer
from utils import load_docno, print_trec_run


def main(X, y, qid,X_test, y_test, qid_test):

    X, y, qid = load_svmlight_file(args.train_file, query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(args.test_file, query_id=True)

    model = CoordinateAscent(n_restarts=1,
                             max_iter=25,
                             scorer=MAPScorer())

    if args.no_validation or args.valid_file == '':
        model.fit(X, y, qid)
    else:
        X_valid, y_valid, qid_valid = load_svmlight_file(args.valid_file, query_id=True)
        model.fit(X, y, qid, X_valid, y_valid, qid_valid)
    pred = model.predict(X_test, qid_test)

    score = MAPScorer(y_test, pred, qid_test)
    print('MAP Score: %s'%score)

    if args.output_file:
        docno = load_docno(args.test_file, letor=True)
        print_trec_run(qid_test, docno, pred, output=open(args.output_file, 'wb'))


if __name__ == '__main__':
    main()
