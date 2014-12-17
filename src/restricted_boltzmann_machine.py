import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images


class RBM(object):
    def __init__(self, 
                 input=None,
                 n_visible=784,
                 n_hidden=500,
                 W=None,
                 hbias=None,
                 vbias=None,
                 numpy_rng=None,
                 theano_rng=None):
        ''' Construct the RBM '''
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # If there is no given numpy range then create random number gen
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # W is uniformly sampled from a symmetric distribution
        if W is None:
            initial_W = np.asarray(numpy_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                                                     high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                                                     size=(n_visible, n_hidden)),
                                   dtype=th.config.floatX)
            W = th.shared(value=initial_W, name='W', borrow=True)
        # Initial the bias for both the hidden and visible units
        if hbias is None:
            hbias = th.shared(value=np.zeros(n_hidden,dtype=th.config.floatX),
                              name='hbias',
                              borrow=True)
        if vbias is None:
            vbias = th.shared(value=np.zeros(n_visible,dtype=th.config.floatX),
                              name='vbias',
                              borrow=True)
        # Initialize input layer
        self.input = input
        if not input:
            self.input = T.matrix('input')
        # Set instance values
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias] # Only put shared variables from this function here

    # Needed for computing the gradient of the parameters
    def free_energy(self, v_sample):
        ''' Compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    # These functions construct the symbolic graph associated with P(h_i = 1|v) = sigm(c_i + W_i v)
    # i.e. probabilistic version of neuron activation function
    def propup(self, vis):
        ''' Propogate the visible units activation upwards to the hidden units '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' Infer state of hidden units given visible units '''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # Sample the hidden units given their activation
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=th.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    # These functions construct the symbolic graph associated with P(v_j = 1|h) = sigm(b_j + W'_j h)
    def propdown(self, hid):
        ''' Propagate the hidden units activation downwards to the visible units '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' Infer state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # Sample the visible units given their activations
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=th.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    # Implement the Gibbs sampling as the transition operator when running Markov chain to convergence
    def gibbs_hvh(self, h0_sample):
        ''' Implement one step of Gibbs sampling starting from hidden state '''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' Implement one step of Gibbs sampling starting from visible state '''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    # Generate the sumbolic gradients for CD-k or PCD-k (Markov Chain convergence techniques) updates
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        ''' Implements one step of CD-k or PCD-k '''
        # Compute the positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # Decide how to initalize persistent chain (depends on CD or PCD choice)
        if persistent is None:
            chain_start = ph_sample # Hidden sample generated during positive phase
        else:
            chain_start = persistent

        # Perform negative phase
        (
         [pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates
        ) = th.scan(self.gibbs_hvh,
                    outputs_info=[None, None, None, None, None, chain_start], # Chain start corresponds to 6th output
                    n_steps=k)
        # Determine gradients on RBM parameters
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # Don't compute the gradient through the Gibbs sampling
        grad_params = T.grad(cost, self.params, consider_constant=[chain_end])
        # Construct the update dictionary
        for grad_param, param in zip(grad_params, self.params):
            updates[param] = param - grad_param * T.cast(lr,dtype=th.config.floatX)
        # PCD specific updating
        if persistent:
            updates[persistent] = nh_samples[-1]
            # Use pseudo-likelihood as a proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # Use reconstruction cross-entropy as a proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        ''' Stochastic approximation to the pseudo-likelihood '''

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = th.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        ''' Approximation to the reconstruction error '''
        
        cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                                     (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                               axis=1))
        return cross_entropy

