from cnn import Net
import utils

#Compute Weighted Linear Regression (LIME with linear regressor
def hat_beta_n(x, net: Net, d, n=10000):

  #get original prediction
  x_out = net(x.view(1,1,28,28))
  x_pred = toValue(x_out)

  #Create an z: (n,d) matrix of bernoulli variables
  z = stats.bernoulli.rvs(0.5, size=(n,d), random_state=2)
  z[0] = np.ones(d)

  #initialise vectors y and pi_i
  y = []
  pii = []

  for i in range(n):
    #trudge original x and forward pass
    xi = trudge_input(x, segments, z[i])
    yi = net(xi.view(1,1,28,28))[0][x_pred].item()

    #Append to vectors pi_i and y
    pii.append(compute_psi(1-sum(z[i].astype(float))/len(z[i])))
    y.append(yi)

  y = np.array(y).reshape(n,1)
  W = np.diag(pii)
  Z = np.concatenate((np.ones(n).reshape(n,1), np.array(z.astype(float))), axis=1)

  hat_beta = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(Z.T, W), Z)), Z.T), W),y)
  return hat_beta

def expected_beta(x, net, d):
  #get original prediction
  x_out = net(x.view(1,1,28,28))
  pred_x = toValue(x_out)

  #Init vars for E(pi_i f(x)) and E(pi_i z_j f(x))
  E_pi_fx = 0
  E_pi_zfx = np.zeros(d)

  #iterate over every binary combination of trudges
  for i in range(2**d):
    #turn i into binary shape with length d
    bit = np.array([int(x) for x in "{0:b}".format(i)])
    zeros = np.zeros(d-len(bit))
    zi = np.concatenate((zeros, bit)).astype(float)

    #trudge original x and forward pass
    xi = trudge_input_mean(x, grid(x, splits=4), zi)
    yi = net(xi.view(1,1,28,28))[0][pred_x].item()
    pii = compute_psi(1-sum(zi.astype(float))/len(zi))
    E_pi_fx += yi*pii
    E_pi_zfx += zi*yi*pii

  #Calculate beta according to Corollary 3
  
  beta0 = (compute_sigma_0(d)*E_pi_fx+compute_sigma_1(d)*sum(E_pi_zfx))
    /(compute_dencst(d)* 2**d)
  betaj = [(compute_sigma_1(d)*E_pi_fx+compute_sigma_2(d)*E_pi_zfx[j]
            +compute_sigma_3(d)*(sum(E_pi_zfx)-E_pi_zfx[j]))/
                (compute_dencst(d)* 2**d) for j in range(d)]
  return np.insert(betaj, 0, beta0).reshape(d+1,1)
  
  #Here I recycled def of sigma, dencst etc. from Garreaus LIME on github
