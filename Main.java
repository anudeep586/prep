import java.util.*;

public class Main {
    public static void main(String[] args) {
      String[] array = new String[10];
      array[0]="23";
      for(int i=0;i<1;i++){
       if(array[i].charAt(0)=='2'){
        System.out.println(true);
       }else{
         System.out.println(false);
       }
      }
  }
}

import java.util.*;

public class Main {
    public static void main(String[] args) {
      String[] array = new String[10];
      array[0]="Anudeep";
      for(int i=0;i<array.length;i++){
        System.out.println(array[0].indexOf("p"));
      }
  }
}

Useful Methods:

add(): Add elements to the ArrayList.
get(int index): Retrieve an element at a specified index.
indexOf(Object o): Find the index of the first occurrence of an element.
contains(Object o): Check if an element exists in the ArrayList.
remove(int index): Remove an element at the specified index.

import java.util.*;

public class Main {
    public static void main(String[] args) {
      ArrayList<String> array = new ArrayList<>();
      array.add("Anudeep");
      System.out.println(array.get(0));      
  }
}

import java.util.*;

public class Main {
    public static void main(String[] args) {
      ArrayList<String> array = new ArrayList<>();
      array.add("Anudeep");
      for(int i=0;i<array.size();i++){
        System.out.println(array.contains("Anudeep"));
      }
      
  }
}

import java.util.*;

public class Main {
    public static void main(String[] args) {
      ArrayList<Integer> array = new ArrayList<>();
      array.add(4);
      for(int i=0;i<array.size();i++){
        System.out.println(array.contains("Anudeep"));
      }
      
  }
}


import java.util.*;

public class Main {
  public static void ContainsOrNot(Integer val,ArrayList<Integer> array){
        System.out.println(array.contains(val));
      }
    public static void main(String[] args) {
      
      ArrayList<Integer> array = new ArrayList<>();
      array.add(4);
      array.add(0);
      array.add(2333);
      for(int i=0;i<array.size();i++){
        ContainsOrNot(77,array);
      }
      
  }
  
}
