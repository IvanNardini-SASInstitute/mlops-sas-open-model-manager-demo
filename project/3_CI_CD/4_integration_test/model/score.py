
import pandas as pd
import joblib
import settings

def score (CLAGE, CLNO, DEBTINC, DELINQ, DEROG, JOB, LOAN, MORTDUE, NINQ, REASON, VALUE, YOJ):
    "Output: P_BAD0, P_BAD1"
    
    try:
       _thisModelFit
    except NameError:
        with open(settings.pickle_path + "rfor_pipeline.pickle", 'rb') as _pFile:
            _thisModelFit = joblib.load(_pFile)

    # Construct the input array for scoring (the first term is for the Intercept)
    input_array = pd.DataFrame([[CLAGE, CLNO, DEBTINC, DELINQ, DEROG, JOB, LOAN, MORTDUE, NINQ, REASON, VALUE, YOJ]],
        columns = ['CLAGE', 'CLNO', 'DEBTINC', 'DELINQ', 'DEROG', 'JOB', 'LOAN', 'MORTDUE', 'NINQ', 'REASON', 'VALUE', 'YOJ'])
#     ,dtype = float

    # Calculate the predicted probabilities
    _predProb = _thisModelFit.predict_proba(input_array)

    # Retrieve the event probability
    P_BAD0 = float(_predProb[0][0])
    P_BAD1 = float(_predProb[0][1])

    return(P_BAD0, P_BAD1)
