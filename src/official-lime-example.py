
#For documentation see 

# LIME_base.py
# LIME_text.py
# LIME_image.py

# at https://github.com/marcotcr/LIME/tree/master/LIME

def OfficialLIME_GPT2(texts):
    token = tokenizer(texts, return_tensors="pt", padding='longest')
    out = model.generate(
            **token, 
            max_new_tokens=1,
            return_dict_in_generate=True, 
            output_scores=True)

    prob_labels = torch.softmax(out["scores"][-1], dim=-1)
    top_label = np.argsort(prob_labels[0])[-1:] #<-- only diff to GPT2
    
    return np.array(prob_labels[:, top_label])






explainer = LIMETextExplainer()

exp = explainer.explain_instance(
        text_instance=sentence,
        classifier_fn=OfficialLIME_GPT2,
        top_labels=1,
        num_features=5,
        num_samples=500,
        model_regressor=LinearRegression(fit_intercept=True))

print(f"Intercept: {exp.intercept}")
print(f"Coefficients: {exp.as_list(label=0)}")
print(f"score: {exp.score}")
