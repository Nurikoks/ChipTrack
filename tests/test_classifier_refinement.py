from app.pipeline.features import NAMES
from app.training import predicted_row


def artifact(bias,positive=3,negative=5):
    return {'types':['A'],'feature_names':NAMES,'mean':[0.]*len(NAMES),'scale':[1.]*len(NAMES),
            'weights':[[0.] for _ in NAMES],'bias':[bias],'threshold':.85,'suppress_threshold':.1,
            'positive_groups':{'A':positive},'negative_groups':{'A':negative}}


def row(preds):
    return {'windows':[{'start':0.,'end':8.,'x':[0.]*len(NAMES)}],
            'preds':preds,'base_preds':preds,'verdict':'violation' if preds else 'no_violation',
            'base_verdict':'violation' if preds else 'no_violation','base_review_required':False}


def test_verified_classifier_can_reject_an_incorrect_rule_event():
    source=row([{'type':'A','t':4.,'conf':.95}])
    result=predicted_row(artifact(-5.),source)
    assert result['preds']==[] and result['verdict']=='no_violation'
    # With too few independent positive or negative examples, it cannot override rules.
    assert predicted_row(artifact(-5.,positive=2),source)['preds']==source['preds']
    assert predicted_row(artifact(-5.,negative=4),source)['preds']==source['preds']


def test_evaluation_counts_rare_class_hypotheses_as_manual_review():
    result=predicted_row(artifact(5.,positive=1),row([]))
    assert result['preds']==[] and result['verdict']=='needs_review'
    result=predicted_row(artifact(5.,positive=3),row([]))
    assert result['verdict']=='violation' and result['preds'][0]['type']=='A'
