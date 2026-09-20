package inc.reactor.cookbook;

import java.awt.Graphics;
import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

final class Preview implements AutoCloseable {
    private final AtomicReference<BufferedImage> latest = new AtomicReference<>();
    private final JFrame frame;
    private final Timer timer;

    private Preview(JFrame frame, Timer timer) {
        this.frame = frame;
        this.timer = timer;
    }

    static Preview open(boolean enabled) throws Exception {
        if (!enabled || GraphicsEnvironment.isHeadless()) {
            return new Preview(null, null);
        }
        Preview[] result = new Preview[1];
        SwingUtilities.invokeAndWait(() -> {
            JFrame window = new JFrame("Reactor Preview");
            Timer repaintTimer = new Timer(42, event -> window.repaint());
            result[0] = new Preview(window, repaintTimer);
            window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            window.setContentPane(new JPanel() {
                private static final long serialVersionUID = 1L;

                @Override
                protected void paintComponent(Graphics graphics) {
                    super.paintComponent(graphics);
                    BufferedImage image = result[0].latest.get();
                    if (image != null) {
                        graphics.drawImage(image, 0, 0, getWidth(), getHeight(), null);
                    }
                }
            });
            window.setSize(960, 540);
            window.setVisible(true);
            repaintTimer.start();
        });
        return result[0];
    }

    void submit(byte[] bgra, int width, int height) {
        latest.set(image(bgra, width, height));
    }

    void save(Path path) throws IOException {
        BufferedImage image = latest.get();
        if (image == null) {
            throw new IOException("no preview frame submitted");
        }
        if (!ImageIO.write(image, "png", path.toFile())) {
            throw new IOException("PNG writer unavailable");
        }
    }

    static BufferedImage image(byte[] bgra, int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        long expected = (long) width * height * 4;
        if (expected != bgra.length) {
            throw new IllegalArgumentException("BGRA length does not match dimensions");
        }
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        int[] pixels = new int[width * height];
        for (int index = 0; index < pixels.length; index++) {
            int at = index * 4;
            int blue = bgra[at] & 0xff;
            int green = bgra[at + 1] & 0xff;
            int red = bgra[at + 2] & 0xff;
            int alpha = bgra[at + 3] & 0xff;
            pixels[index] = alpha << 24 | red << 16 | green << 8 | blue;
        }
        image.setRGB(0, 0, width, height, pixels, 0, width);
        return image;
    }

    @Override
    public void close() {
        if (frame == null) {
            return;
        }
        Runnable dispose = () -> {
            timer.stop();
            frame.dispose();
        };
        if (SwingUtilities.isEventDispatchThread()) {
            dispose.run();
        } else {
            SwingUtilities.invokeLater(dispose);
        }
    }
}
