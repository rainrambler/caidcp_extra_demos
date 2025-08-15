public class DeadlockDemo {

    // 创建两个锁对象
    private static final Object lock1 = new Object();
    private static final Object lock2 = new Object();

    public static void main(String[] args) {
        // 线程1：先获取lock1，再获取lock2
        Thread thread1 = new Thread(() -> {
            synchronized (lock1) {
                System.out.println("线程1获取了lock1");
                try {
                    // 睡眠一小会儿，确保线程2能获取到lock2
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                // 死锁发生点1：线程1持有lock1，尝试获取lock2
                synchronized (lock2) {
                    System.out.println("线程1获取了lock2");
                }
            }
        });

        // 线程2：先获取lock2，再获取lock1
        Thread thread2 = new Thread(() -> {
            synchronized (lock2) {
                System.out.println("线程2获取了lock2");
                try {
                    // 睡眠一小会儿，确保线程1能获取到lock1
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                // 死锁发生点2：线程2持有lock2，尝试获取lock1
                synchronized (lock1) {
                    System.out.println("线程2获取了lock1");
                }
            }
        });

        thread1.start();
        thread2.start();

        try {
            // 等待线程结束
            thread1.join();
            thread2.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("程序结束"); // 这一行永远不会执行，因为发生了死锁
    }
}