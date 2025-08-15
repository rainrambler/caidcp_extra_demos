public class FixedDeadlockDemo {

    private static final Object lock1 = new Object();
    private static final Object lock2 = new Object();

    public static void main(String[] args) {
        // 两个线程都按照lock1 -> lock2的顺序获取锁
        Thread thread1 = new Thread(() -> {
            synchronized (lock1) {
                System.out.println("线程1获取了lock1");
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (lock2) {
                    System.out.println("线程1获取了lock2");
                }
            }
        });

        Thread thread2 = new Thread(() -> {
            synchronized (lock1) {
                System.out.println("线程2获取了lock1");
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (lock2) {
                    System.out.println("线程2获取了lock2");
                }
            }
        });

        thread1.start();
        thread2.start();

        try {
            thread1.join();
            thread2.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("程序正常结束"); // 现在这一行可以执行了
    }
}