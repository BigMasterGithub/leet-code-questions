package leetcodes.juc;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author 张壮
 * @description 三个线程交替打印ABC
 * @since 2023/9/5 10:46
 **/

public class Solution {

    public static void main(String[] args) {

        WaitNotify work2 = new WaitNotify(1, 5);
        new Thread(() -> {
            work2.print("a", 1, 2);
        }).start();
        new Thread(() -> {
            work2.print("b", 2, 3);
        }).start();
        new Thread(() -> {
            work2.print("c", 3, 1);
        }).start();
    }
}
///三个线程交替打印ABC
//法1 wait notify
class WaitNotify {
    private int flag;
    private int count;

    WaitNotify(int flag, int count) {
        this.flag = flag;
        this.count = count;
    }

    public void print(String str, int waitFlag, int nextFlag) {
        for (int i = 0; i < count; i++) {
            synchronized (this) {
                while (this.flag != waitFlag) {
                    try {
                        this.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                System.out.print(str+" ");
                flag = nextFlag;
                this.notifyAll();
            }

        }
    }
}

//法2 await signal
class AwaitSignal extends ReentrantLock {
    private int loopNumber;

    public AwaitSignal(int loopNumber) {
        this.loopNumber = loopNumber;
    }

    public static void main(String[] args) throws InterruptedException {
        AwaitSignal awaitSignal = new AwaitSignal(5);
        Condition condition_a = awaitSignal.newCondition();
        Condition condition_b = awaitSignal.newCondition();
        Condition condition_c = awaitSignal.newCondition();
        new Thread(() -> {
            awaitSignal.print("a", condition_a, condition_b);
        }).start();
        new Thread(() -> {
            awaitSignal.print("b", condition_b, condition_c);
        }).start();
        new Thread(() -> {
            awaitSignal.print("c", condition_c, condition_a);
        }).start();
        Thread.sleep(1_000);
        System.out.println("开始！");
        awaitSignal.lock();
        condition_a.signal();
        awaitSignal.unlock();
    }

    public void print(String str, Condition current, Condition next) {

        for (int i = 0; i < loopNumber; i++) {
            lock();
            try {
                current.await();
                System.out.println(str);
                next.signal();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                unlock();
            }

        }

    }
}

//法3 park unpark
class ParkUnPark {
    private int loopNumber;

    public ParkUnPark(int loopNumber) {
        this.loopNumber = loopNumber;
    }


    public void print(String str, Thread next) {
        for (int i = 0; i < loopNumber; i++) {
            LockSupport.park();
            System.out.println(str);
            LockSupport.unpark(next);
        }
    }

    static Thread t1;
    static Thread t2;
    static Thread t3;

    public static void main(String[] args) {


        ParkUnPark park = new ParkUnPark(5);
        t1 = new Thread(() -> {
            park.print("a", t2);
        });
        t2 = new Thread(() -> {
            park.print("b", t3);
        });
        t3 = new Thread(() -> {
            park.print("c", t1);
        });
        t1.start();
        t2.start();
        t3.start();
        LockSupport.unpark(t1);
    }
}