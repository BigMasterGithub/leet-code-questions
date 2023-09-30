package written.examination.huawei_9_27;

import java.util.*;

public class Main2 {
    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        String line1 = in.nextLine();
        line1=line1.substring(1, line1.lastIndexOf(']'));
        for (String string : line1.split(" ")) {
            list1.add(Integer.valueOf(string));
        }


        String line2 = in.nextLine();
      line2=  line2.substring(1, line2.lastIndexOf(']'));

        for (String string : line2.split(" ")) {
            list2.add(Integer.valueOf(string));
        }
        String operation = in.next();
//        list1.add(3);
//        list1.add(0);
//        list1.add(6);
//        list1.add(0);
//        list2.add(9);
//        list2.add(0);
//        list2.add(0);
        List<Integer> ans = fun_huawei2(list1, list2, operation);
//        int ans = fun_huawei(new int[]{9,4,5,2,4});
        String startChar="[";
        String endChar="]";
        StringBuilder finalans= new StringBuilder();
        ans.stream().forEach((cur)->{
            finalans.append(cur+" ");
        });

        System.out.println(  startChar+finalans.toString().trim()+endChar);
    }

    private static List<Integer> fun_huawei2(List<Integer> list1, List<Integer> list2, String operation) {
        int size = list1.size();
        HashMap<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < size; i++) {
            map1.put(size - i - 1, list1.get(i));
        }

        int size2 = list2.size();
        HashMap<Integer, Integer> map2 = new HashMap<>();
        for (int i = 0; i < size2; i++) {
            map2.put(size2 - i - 1, list2.get(i));
        }
        HashMap<Integer, Integer> ans = new HashMap<>();
        switch (operation) {
            case "*": {
                for (int i = 0; i < map1.size(); i++) {
                    int a = map1.get(i);
                    for (int j = 0; j < map2.size(); j++) {
                        int b = map2.get(j);
                        int curAns = a * b;
                        ans.put(i + j, ans.getOrDefault(i + j, 0) + curAns);
                    }
                }
                break;
            }
            case "+": {
                int maxSize = Math.max(map1.size(), map2.size());
                int index = 0;
                while (index < maxSize) {

                    int i = map1.getOrDefault(index, 0) + map2.getOrDefault(index, 0);
                    ans.put(index, i);
                    index++;
                }

                break;

            }

            case "-":
                int maxSize = Math.max(map1.size(), map2.size());
                int index = 0;
                while (index < maxSize) {

                    int i = map1.getOrDefault(index, 0) - map2.getOrDefault(index, 0);
                    ans.put(index, i);
                    index++;
                }

                break;
            default:
                break;
        }
        List<Integer> finalAns = new ArrayList<>();
        for (int i = ans.size() - 1; i >= 0; i--) {

            finalAns.add(ans.getOrDefault(i, 0));
        }

        return finalAns.size() == 0 ? Arrays.asList(0) : finalAns;

    }
}

