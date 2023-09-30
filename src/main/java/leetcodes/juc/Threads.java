package leetcodes.juc;

import java.util.function.IntConsumer;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/28 15:26
 **/
public class Threads {


    public static void main(String[] args) throws InterruptedException {
       /* FooBar fooBar = new FooBar(5);
        new Thread(() -> {
            try {
                fooBar.foo(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }).start();
        new Thread(() -> {
            try {
                fooBar.bar(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }).start();*/


     /*   FizzBuzz fizzBuzz = new FizzBuzz(15);
        new Thread(() -> {
            try {
                fizzBuzz.fizz(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "A").start();

        new Thread(() -> {
            try {
                fizzBuzz.buzz(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "B").start();

        new Thread(() -> {
            try {
                fizzBuzz.fizzbuzz(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "C").start();

        new Thread(() -> {
            try {
                fizzBuzz.number(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "D").start();
*/
        Foo foo = new Foo();


        new Thread(() -> {
            try {
                foo.second(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "B").start();
        Thread.sleep(1000);
        new Thread(() -> {
            try {
                foo.third(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "C").start();
        Thread.sleep(1000);
        new Thread(() -> {
            try {
                foo.first(null);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, "A").start();
    }


}

//1114.按序打印 三个线程按顺序输出 one ，two ，three
class Foo {
    private int flag;
    private Object lock = new Object();

    public Foo() {
        flag = 1;
    }


    public void second(Runnable printSecond) throws InterruptedException {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + "获取了锁，此时flag为" + flag);
            while (flag != 2) {
                System.out.println(Thread.currentThread().getName() + "进入等待中");
                lock.wait();
            }
            System.out.println("被唤醒了");
            System.out.println("two");
            flag = 2;
            lock.notifyAll();
        }


    }

    public void third(Runnable printThird) throws InterruptedException {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + "获取了锁，此时flag为" + flag);

            while (flag != 3) {
                System.out.println(Thread.currentThread().getName() + "进入等待中");

                lock.wait();
            }
            System.out.println("three");

            flag = 6;
            lock.notifyAll();
        }

    }

    public void first(Runnable printFirst) throws InterruptedException {
        synchronized (lock) {
            System.out.println(Thread.currentThread().getName() + "获取了锁，此时flag为" + flag);

            while (flag != 1) {
                System.out.println(Thread.currentThread().getName() + "进入等待中");

                lock.wait();
            }
            System.out.println("one");
            flag = 6;
            lock.notifyAll();
        }


    }
}

//1195.交替打印字符串 -- 不需要控制顺序，谁满足谁打印
class FizzBuzz {
    private int n;
    private int flag;
    private Object lock = new Object();

    public FizzBuzz(int n) {
        this.n = n;
        flag = 1;
    }

    // printFizz.run() outputs "fizz".
    public void fizz(Runnable printFizz) throws InterruptedException {
        while (flag <= n) {
            synchronized (lock) {
                while (flag <= n && flag % 3 == 0) {
                    System.out.println(flag + "->fizz");
                    flag++;
                }

            }


        }


    }

    // printBuzz.run() outputs "buzz".
    public void buzz(Runnable printBuzz) throws InterruptedException {
        while (flag <= n) {
            synchronized (lock) {
                while (flag <= n && flag % 3 != 0 && flag % 5 == 0) {
                    System.out.println(flag + "->buzz");
                    flag++;
                }
            }
        }
    }

    // printFizzBuzz.run() outputs "fizzbuzz".
    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
        while (flag <= n) {
            synchronized (lock) {
                while (flag <= n && flag % 3 == 0 && flag % 5 == 0) {
                    System.out.println(flag + "->fizzbuzz");
                    flag++;
                }
            }
        }
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void number(IntConsumer printNumber) throws InterruptedException {
        while (flag <= n) {
            synchronized (lock) {
                while (flag <= n && flag % 3 != 0 && flag % 5 != 0) {
                    System.out.println(flag);
                    flag++;
                }
            }
        }
    }
}

//1115.交替打印FooBar
class FooBar {
    private int n;

    public FooBar(int n) {
        this.n = n;
        flag = 1;
    }

    private int flag;
    private Object lock = new Object();

    public void foo(Runnable printFoo) throws InterruptedException {

        for (int i = 0; i < n; i++) {
            synchronized (lock) {
                if (flag != 1) lock.wait();
                flag = 2;
                // printFoo.run() outputs "foo". Do not change or remove this line.
//                printFoo.run();
                System.out.println("foo");
                lock.notifyAll();
            }

        }
    }

    public void bar(Runnable printBar) throws InterruptedException {

        for (int i = 0; i < n; i++) {
            synchronized (lock) {
                if (flag != 2) lock.wait();
                flag = 1;
                // printBar.run() outputs "bar". Do not change or remove this line.
                System.out.println("bar");
                lock.notifyAll();
            }
        }
    }
}
