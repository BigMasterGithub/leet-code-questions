package written.examination.xiaomi_9_23;

import java.lang.reflect.Array;
import java.util.*;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/23 17:01
 **/
public class Main2 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        int numCourses = in.nextInt();
        in.nextLine();

        int[][] prerequisites = new int[numCourses][2];

        String line = in.nextLine();
        String[] split = line.split(",");
//        System.out.println(Arrays.toString(split));
        for (int i = 0; i < split.length; i++) {
            prerequisites[i][0] = Integer.valueOf(split[i].charAt(0) + "");

            prerequisites[i][1] = Integer.valueOf(split[i].charAt(2) + "");

        }
        for (int i = 0; i < numCourses; i++) {
            System.out.println(prerequisites[i][0] +","+ prerequisites[i][1]);
        }
        LinkedList<Integer> queue = new LinkedList();
        boolean ans = false;
        List<List<Integer>> adjacencyList = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adjacencyList.add(new ArrayList<>());

        }


        int[] indegree = new int[numCourses];
        for (int i = 0; i < prerequisites.length; i++) {

            int curCourseIndex = prerequisites[i][0];
            int prerequisiteCourseIndex = prerequisites[i][1];
            adjacencyList.get(prerequisiteCourseIndex).add(curCourseIndex);

            indegree[curCourseIndex]++;
        }

        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) queue.offer(i);

        }
        while (!queue.isEmpty()) {
            Integer curprerequisiteCourseIndex = queue.poll();
            numCourses--;
            int size = adjacencyList.get(curprerequisiteCourseIndex).size();
            for (int i = 0; i < size; i++) {
                int courseIndex = adjacencyList.get(curprerequisiteCourseIndex).get(i);
                indegree[courseIndex]--;
                if (indegree[courseIndex] == 0) queue.add(courseIndex);
            }


        }
        ans = numCourses == 0;
        System.out.println(ans ? 1 : 0);
    }
}

