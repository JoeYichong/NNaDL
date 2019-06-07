package yich.nn.single;

import yich.nn.math.NNMath;

/**
 * Sigmoid neuron using cross-entropy cost function
 * Input value is 1.
 * Desired output value is 0.
 *
 * */

public class SingleSigmoidNeuronTest2 extends SingleNeuron {
    public SingleSigmoidNeuronTest2() {
        setWeight(2.4);
        setBias(2.6);
        setX(1);
        setY(0);
        setEta(2);
        setEpoch(50);
    }

    @Override
    public double nabla_w() {
        double z = weight * x + bias;
        return (NNMath.sigmoid(z) - y) * x;
    }

    @Override
    public double nabla_b() {
        double z = weight * x + bias;
        return NNMath.sigmoid(z) - y;
    }

    @Override
    public double activate(double z) {
        return NNMath.sigmoid(z); // linear neuron
    }


    @Override
    public double cost() {
        return NNMath.cross_entropy(feedforward(), y);
    }


    public static void main(String[] args) {
        new SingleSigmoidNeuronTest2().run();
    }

}
