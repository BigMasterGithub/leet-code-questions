package written.examination.cainiao_9_21;


import java.util.HashMap;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/21 16:20
 **/
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int x = in.nextInt();
        int y = in.nextInt();
        in.nextLine();
        String s = in.nextLine();
        HashMap<Character, Integer> map = new HashMap<>();
        char[] charArray = s.toCharArray();
        for (int i = 0; i < charArray.length; i++) {
            if (charArray[i] == 'A') {
                map.put('A', map.getOrDefault('A', 0) + 1);
            } else if (charArray[i] == 'B') {
                map.put('B', map.getOrDefault('B', 0) + 1);
            } else {
                map.put('C', map.getOrDefault('C', 0) + 1);
            }
        }
        int ans = 0;
        int sumA = map.getOrDefault('A',0);
        int sumB = map.getOrDefault('B',0);
        int sumC = map.getOrDefault('C',0);

        System.out.println(sumA);
        System.out.println(sumB);
        System.out.println(sumC);
        if(sumB==0){
            System.out.println(0);
            return ;
        }
        if(sumA == 0){
            System.out.println(Math.min(sumB,sumC)*y);
            return;
        } if(sumC ==0){
            System.out.println(Math.min(sumB,sumA)*x);
            return;
        }

        if (x > y) {

            int countAB = Math.min(sumA, sumB);

            ans += countAB * x;
            if (sumB > sumA) {
                map.put('B', sumB - sumA);
            } else {
                System.out.println(ans);
                return;
            }
            sumB = map.get('B');
            int countBC = Math.min(sumB, sumC);

            ans += countBC * y;

            System.out.println(ans);
        } else {
            int countBC = Math.min(sumC, sumB);

            ans += countBC * y;
            if (sumB > sumC) {
                map.put('B', sumB - sumC);
            } else {
                System.out.println(ans);
                return;
            }
            sumB = map.get('B');
            int countAB = Math.min(sumB, sumA);

            ans += countAB * x;

            System.out.println(ans);
        }

    }
}

