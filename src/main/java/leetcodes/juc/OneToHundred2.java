package leetcodes.juc;

//两个线程交替打印1~100数字
public class OneToHundred2 {
    public static void main(String[] args){
        Number number = new Number();
        Thread t1 = new Thread(number);
        Thread t2 = new Thread(number);
        t1.setName("线程1");
        t2.setName("线程2");
        t1.start();
        t2.start();
    }
}

class Number implements Runnable{
    private int number = 1;

    @Override
    public void run(){
        while(number < 100){
            synchronized(this){
                System.out.println(Thread.currentThread().getName()+"->"+number);
                number++;
                notify();
                try{
                    if(number < 100){
                        wait();
                    }
                }catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}