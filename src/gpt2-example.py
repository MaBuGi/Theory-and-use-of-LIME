
#Propagate tokens through GPT2 model
def GPT_prob(token, model):
  out = model.generate(**token, max_new_tokens=1, 
  return_dict_in_generate=True, output_scores=True)

  prob_labels = torch.softmax(out["scores"][-1], dim=-1)
  top_label = np.argsort(prob_labels[0])[-1]
  name_top_label= tokenizer.decode(top_label)

  print(f"Top label:{name_top_label} \nTop P: {round(prob_labels[0, 
  top_label].item(), 3)}\n")

  return prob_labels[:, top_label]

#Compute psi values for weights
def compute_psi(t,nu=0.25):
  return np.exp(-np.square(1.0-np.sqrt(1.0 - t))/(2*nu**2))

#Compute weights for regression
def pi(Z):
  return compute_psi(np.subtract(np.ones(Z.shape[0]), np.sum(Z, 
                        axis=1)/Z.shape[1]))

#Print results of regression
def print_results(feature_names, regressor, Z, y, weights,          
                    interaction=False):
  rounded = [round(c, 5) for c in regressor.coef_]
  coefs = dict(zip(feature_names, rounded))

  print(f"Intercept: {round(regressor.intercept_, 5)})
  
  print(f"Coefficients: {coefs}")
  print(f"Weighted R^2: {round(regressor.score(Z, y,
        sample_weight=weights),5)}")
  return

#Run LIME on a language model

def LIME_TEXT(model, token, feature_names, Z, model_regressor, 
                interaction=False):
  y = GPT_prob(token, model)
  weights = pi(Z)

  if interaction:
    Z = append_interaction_columns(Z)
  
  Z = Z.astype(float)

  reg = model_regressor.fit(Z, y, sample_weight=weights)
  print_results(feature_names, reg, Z, y, weights)
  return


#EXAMPLE 1: RANDOM DESIGN Z, Attention LIME
Z = stats.bernoulli.rvs(0.5, size=(n,d), random_state=7)
Z[0] = np.ones(d)

token = tokenizer([sentence]*n, return_tensors="pt")
token["attention_mask"] = torch.from_numpy(Z)

#Linear Regression (no interaction)
LIME_TEXT(model, token, words, Z, LinearRegression(fit_intercept=True))

