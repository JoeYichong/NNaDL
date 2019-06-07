package yich.nn.single;

import yich.nn.plot.NNChart;

abstract public class SingleNeuron {
    double weight = 0.8;
    double bias = 0.8;
    double x = 1;
    double y = 0;
    double eta = 2;
    int epoch = 30;

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getEta() {
        return eta;
    }

    public void setEta(double eta) {
        this.eta = eta;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    abstract public double nabla_w();

    abstract public double nabla_b();

    abstract public double activate(double z);

    public double feedforward(){
        return activate(weight * x + bias);
    }

    public void update() {
        weight -= nabla_w() * eta;
        bias -= nabla_b() * eta;
        print();
    }

    abstract public double cost();

    public void print() {
        System.out.println("x: " + getX());
        System.out.println("weight: " + getWeight());
        System.out.println("bias: " + getBias());
        System.out.println("output: " + feedforward());
        System.out.println("cost: " + cost());
        System.out.println();
    }

    public void run() {
        print();
        double[] costs = new double[epoch];
        for (int i = 0; i < epoch; i++) {
            System.out.println("update-" + i + ": ");
            update();
            costs[i] = cost();
        }
        new NNChart().plot(costs);
    }
}
