package written.examination.jingdong_9_16;


import java.util.*;

public class Main2 {

    private static boolean visited[];
    private static ArrayList<ArrayList<Integer>> adj;

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int round = in.nextInt();
        while (round-- > 0) {
            int N = in.nextInt();
            int target = in.nextInt();
            adj = new ArrayList<>();
            visited = new boolean[N + 1];
            for (int i = 0; i <= N; i++) {
                adj.add(new ArrayList<>());
                visited[i] = false;
            }
            for (int i = 0; i < N - 1; i++) {
                int u = in.nextInt();
                int v = in.nextInt();
                adj.get(u).add(v);
                adj.get(v).add(u);

            }
            boolean ans = fun(target);
            if (ans) {
                System.out.println("win");
            } else {
                System.out.println("lose");
            }
        }
    }

    private static boolean fun(int target) {
        visited[target] = true;
        int count = 0;
        for (int neighboor : adj.get(target)) {
            if(!visited[neighboor]){
                if (fun(neighboor)) {
                    count++;
                }
            }

        }
        if (count == 0) {
            return true;
        } else if (count == 1 && target != 1) {
            return true;
        } else {
            return false;
        }
    }
}
/*
1
5 2
1 2
1 3
2 4
2 5*/