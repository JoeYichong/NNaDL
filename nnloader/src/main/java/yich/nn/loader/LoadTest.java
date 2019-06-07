package yich.nn.loader;

public class LoadTest {

    public static void main(String[] args) {
        IDXObject obj = IDXFileParser.parse(DataPath.MNIST.getProperty("test.images"));
        IDXObject obj2 = IDXFileParser.parse(DataPath.MNIST.getProperty("test.labels"));
        int index = 1157;

        obj.printInfo();
        MNISTParser.printImage(obj, index);
//        MNISTParser.saveImage(obj.get(index), "D:\\TEMP\\A\\test.bmp");
        MNISTParser.displayImage(obj.get(index));

        obj2.printInfo();
        System.out.println("Y: " + obj2.data[index]);


    }
}
