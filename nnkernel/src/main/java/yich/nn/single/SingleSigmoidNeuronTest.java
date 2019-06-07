package yich.nn.single;

import yich.nn.math.NNMath;

/**
 * Sigmoid neuron using quadratic cost function
 * Input value is 1.
 * Desired output value is 0.
 *
 * */
public class SingleSigmoidNeuronTest extends SingleNeuron {

    public SingleSigmoidNeuronTest() {
        setWeight(2.6);
        setBias(2.4);
        setX(1);
        setY(0);
        setEta(2);
        setEpoch(50);
    }

    @Override
    public double nabla_w() {
        double z = weight * x + bias;
        return (NNMath.sigmoid(z) - y) * NNMath.sigmoid_prime(z) * x;
    }

    @Override
    public double nabla_b() {
        double z = weight * x + bias;
        return (NNMath.sigmoid(z) - y) * NNMath.sigmoid_prime(z);
    }

    @Override
    public double activate(double z) {
        return NNMath.sigmoid(z); // sigmoid neuron
    }


    @Override
    public double cost() {
        return NNMath.quadratic_cost(feedforward(), y);
    }


    public static void main(String[] args) {
        new SingleSigmoidNeuronTest().run();
    }

}
