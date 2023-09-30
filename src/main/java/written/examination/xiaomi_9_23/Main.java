package written.examination.xiaomi_9_23;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/23 16:23
 **/
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        String line_1 = in.nextLine();
        String[] split = line_1.split(",");
        int len = Integer.valueOf(split[0]);
        int radius = Integer.valueOf(split[1]);
        ArrayList<Towers> towers = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            String line = in.nextLine();
            String[] split1 = line.split(",");
            int x = Integer.valueOf(split1[0]);
            int y = Integer.valueOf(split1[1]);
            int power = Integer.valueOf(split1[2]);
            Towers towers1 = new Towers(x, y, power);
            towers.add(towers1);
        }
//        System.out.println(towers);
//        System.out.println("radius=" + radius);
        int[] ans = fun(radius, towers);
        System.out.println(ans[0] + "," + ans[1]);

    }

    private static int[] fun(int radius, ArrayList<Towers> towers) {
        towers.sort((Towers t1, Towers t2) -> {
            return t1.x - t2.x;
        });
        int max_x = towers.get(towers.size() - 1).x + radius;
        towers.sort((Towers t1, Towers t2) -> {
            return t1.y - t2.y;
        });

        int max_y = towers.get(towers.size() - 1).y + radius;
        int[] ans = new int[2];
        int curMax = 0;
        for (int i = 0; i < max_x; i++) {
            for (int j = 0; j < max_y; j++) {

                int p = getPowers(i, j, towers, radius);
//                System.out.println(i + "," + j + " 的强度和为" + p);
                if (p != -1) {
                    if (p > curMax) {
                        ans[0] = i;
                        ans[1] = j;
                        curMax = p;
                    }
                }
            }

        }

        return ans;
    }

    private static int getPowers(int x, int y, ArrayList<Towers> towers, int radius) {
        int ans = 0;
        for (Towers cur : towers) {
            int x1 = cur.x;
            int y1 = cur.y;
            int d = getDistance(x, y, x1, y1);
//            System.out.println("(" + x + "," + y + ") - (" + x1 + "," + y1 + ") =" + d);
            if (d <= radius) {
//                System.out.println("信号强度为：" + cur.p / (1 + d));
                ans += cur.p / (1 + d);
            } else {
                return -1;
            }

        }
        return ans;
    }

    private static int getDistance(int x, int y, int x2, int y2) {
        return (int) Math.sqrt(Math.pow(x - x2, 2) + Math.pow(y - y2, 2));
    }
}

class Towers {
    public int x;
    public int y;
    public int p;

    public Towers(int x, int y, int p) {
        this.x = x;
        this.y = y;
        this.p = p;
    }

    @Override
    public String toString() {
        return "Towers{" +
                "x=" + x +
                ", y=" + y +
                ", p=" + p +
                '}';
    }
}


