package written.examination.xiecheng_9_21.xiecheng_9_21;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/21 19:28
 **/
public class Main3 {
    static boolean[][] visited;

    public static void main(String[] args) {
//        Scanner in = new Scanner(System.in);
//        int rowLen = in.nextInt();
//        int colLen = in.nextInt();
//        in.nextLine();
//        char[][] graph = new char[rowLen][colLen];
//        for (int i = 0; i < rowLen; i++) {
//            char[] charArray = in.nextLine().toCharArray();
//            graph[i] = charArray;
//        }
//        for (int i = 0; i < rowLen; i++) {
//            System.out.println(Arrays.toString(graph[i]));
//        }
//        visited = new boolean[rowLen][colLen];
        int rowLen =2;
        int colLen =3;
        visited = new boolean[rowLen][colLen];
        char[][] graph=new char[][]{{'a','a','d'},{'a','b','c'}};
        int ans = 0;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                visited = new boolean[rowLen][colLen];
                Set<Character> curGrid = new HashSet<>();
                dfs(graph, i, j, curGrid);
                ans += curGrid.size();
            }
        }

        System.out.println(ans);
    }

    private static void dfs(char[][] grid, int i, int j, Set<Character> curGrid) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        if (i < 0 || j < 0 || i >= rowLen || j >= cowLen) return;
        if (visited[i][j]) return;
        visited[i][j] = true;


        if (curGrid.contains(grid[i][j])) return;
        curGrid.add(grid[i][j]);

        dfs(grid, i - 1, j, curGrid);
        dfs(grid, i + 1, j, curGrid);
        dfs(grid, i, j - 1, curGrid);
        dfs(grid, i, j + 1, curGrid);

    }
}

