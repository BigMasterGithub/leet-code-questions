package leetcodes;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/4 17:41
 **/
public class Other {
    //1.求两个整数的最大公约数
    // 默认a小于b
    public int gcd(int a, int b) {
        while (b != 0) {
            int next = a % b;
            a = b;
            b = next;
        }
        return a;

    }

    //2.求两个整数的最小公倍数
    public int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    public static void main(String[] args) {
        System.out.println(new Other().gcd(319, 377));
        System.out.println(new Other().lcm(319, 377));
    }

}

