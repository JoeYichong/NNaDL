package yich.nn.plot;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class NNChart implements Plotter{

    @Override
    public void plot(double[] costs, String x, String y) {
        XYChart chart = new XYChartBuilder().xAxisTitle(x).yAxisTitle(y).width(600).height(400).build();
//        chart.getStyler().setYAxisMin((double) 0);
//        chart.getStyler().setYAxisMax((double) 1);
        XYSeries series = chart.addSeries(x+"-"+y, null, costs);
        series.setMarker(SeriesMarkers.NONE);
        new SwingWrapper(chart).displayChart();
    }

    @Override
    public void plot(double[] costs) {
        plot(costs, "Epoch", "Cost");
    }

}
