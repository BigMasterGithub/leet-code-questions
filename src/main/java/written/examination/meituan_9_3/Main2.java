package written.examination.meituan_9_3;

import java.util.*;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/2 19:14
 **/
public class Main2 {
    public static void main(String[] args) {
       /* Scanner in = new Scanner(System.in);
        int rowLen = in.nextInt();
        int colLen = in.nextInt();

        in.nextLine();

        for (int i = 0; i < rowLen; i++) {
            String curRowData = in.nextLine();
            grpah.add(curRowData);
        }*/

        List<String> graph = new ArrayList<>();
        graph.add("nm");
        graph.add("ex");
        graph.add("ti");
        graph.add("td");
        graph.add("ul");
        graph.add("qu");
        graph.add("ac");
        graph.add("hj");
//        System.out.println(grpah);


       /* if (rowLen < 7) {
            System.out.println("NO");

        return;
    }*/

        String target = "meituan";

        boolean ans = fun(graph, target, 0, 0);
        if (ans) {
            System.out.println("YES");
        } else {
            System.out.println("NO");
        }


    }

    private static boolean fun(List<String> grpah, String target, int curRowIndex, int curTargetIndex) {
        if (curTargetIndex == target.length()) return true;
        if (curRowIndex == grpah.size()) return false;
        //看当前行能不能匹配上
        String strings = grpah.get(curRowIndex);
        if (strings.indexOf(target.charAt(curTargetIndex)) != -1) {
            boolean result = fun(grpah, target, curRowIndex + 1, curTargetIndex + 1);
            if (result) return true;
        } else {
            //不要这行
            boolean result = fun(grpah, target, curRowIndex + 1, curTargetIndex);
            if (result) return true;
        }

        return false;

    }
}

