package yich.nn.single;

import yich.nn.math.NNMath;

/**
 * Linear neuron(y = x)
 * Input value is 1.
 * Desired output value is 0.
 * */
public class SingleLinearNeuronTest extends SingleNeuron{

    public SingleLinearNeuronTest() {
        setWeight(0.93);
        setBias(0.82);
        setX(1);
        setY(0);
        setEta(0.1);
        setEpoch(30);
    }

    @Override
    public double nabla_w() {
        return (weight * x + bias - y) * x;
    }

    @Override
    public double nabla_b() {
        return (weight * x + bias - y);
    }

    @Override
    public double activate(double z) {
        return z; // linear neuron
    }

    @Override
    public double cost() {
        return NNMath.quadratic_cost(feedforward(), y);
    }


    public static void main(String[] args) {
        new SingleLinearNeuronTest().run();
    }

}
