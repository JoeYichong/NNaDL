package yich.nn.loader;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class MNISTParser {

    public static void printImage(IDXObject obj, int index) {
//        byte[] arr = obj.get(index);
//        for (int i = 0; i < arr.length; i++) {
//            if (arr[i] != 0) {
//                System.out.print("@");
//            } else {
//                System.out.print("-");
//            }
//            if ((i + 1) % 28 == 0) {
//                System.out.println();
//            }
//        }

        byte[] data = obj.getData();
        int offset = 784 * index;
        for (int i = 0; i < 784; i++) {
            if (data[i + offset] != 0) {
                System.out.print("@");
            } else {
                System.out.print("-");
            }
            if ((i + 1) % 28 == 0) {
                System.out.println();
            }
        }

    }



    public static int getLabelNum(IDXObject obj, int index) {
        return obj.data[index];
    }

    public static BufferedImage createImage(byte[] arr) {
        final BufferedImage res = new BufferedImage( 28, 28, BufferedImage.TYPE_BYTE_GRAY);
        byte [] imageData = ((DataBufferByte)res.getRaster().getDataBuffer()).getData();
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (byte) (((byte) 255) - arr[i]);
        }
//        for (int x = 0; x < 28; x++) {
//            for (int y = 0; y < 28; y++) {
//                res.setRGB(x, y, ((byte)255) - arr[x + y * 28]);
//            }
//        }
        return res;
    }

    public static void saveImage(byte[] arr, String path) throws IOException {
        final BufferedImage res = createImage(arr);
        ImageIO.write(res, "bmp", new File(path));
    }

    public static void displayImage(byte[] arr) {
        final BufferedImage img = createImage(arr);

        JLabel jLabel = new JLabel(new ImageIcon(img));
        JPanel jPanel = new JPanel();
        jPanel.add(jLabel);
        JFrame frame = new JFrame();
        frame.setTitle("Digit Image");
        frame.add(jPanel);
        //frame.pack();
        frame.setSize(new Dimension(100,100));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }


}
