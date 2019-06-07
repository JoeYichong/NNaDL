package yich.nn.plot;

public interface Plotter {

    void plot(double[] costs, String x, String y);

    void plot(double[] costs);
}
