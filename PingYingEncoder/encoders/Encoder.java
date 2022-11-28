package PingYingEncoder.encoders;

import java.io.*;
import java.util.HashMap;

public class Encoder implements Serializable {
    private HashMap<Character, String> encodingTable = new HashMap<Character, String>();
    private HashMap<String, Character> decodingTable = new HashMap<String, Character>();

    public void setTables(HashMap<Character, String> table) {
        encodingTable = table;
        for (Character key : encodingTable.keySet()) {
            decodingTable.put(encodingTable.get(key), key);
        }
    }

    public String encode(String inputString) {
        inputString = inputString.toLowerCase();
        String outputString = new String();

        for (int index = 0; index < inputString.length() - 1; index++) {
            char c = inputString.charAt(index);
            outputString += encodingTable.get(c) + " ";
        }
        char c = inputString.charAt(inputString.length() - 1);
        outputString += encodingTable.get(c);
        return outputString;
    }

    public String decode(String inputString) {

        String outputString = new String();
        String word = new String();
        for (int index = 0; index < inputString.length(); index++) {
            char c = inputString.charAt(index);
            if (c == ' ') {
                outputString += decodingTable.get(word);
                word = new String();
            } else {
                word += c;
            }
        }
        outputString += decodingTable.get(word);
        return outputString;
    }

    public String toString() {
        String output = new String();
        for (Character key : encodingTable.keySet()) {
            output += key + ": " + encodingTable.get(key) + "\n";
        }
        return output;
    }

    public void save(String path) {
        try {
            FileOutputStream fos = new FileOutputStream(path);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(this);
            oos.close();
            fos.close();
        } catch (Exception e) {
            System.out.println("Error:" + e);
        }
    }

    public void load(String path) {
        try {
            FileInputStream fis = new FileInputStream(path);
            ObjectInputStream ois = new ObjectInputStream(fis);
            Encoder temp = (Encoder) ois.readObject();
            encodingTable = temp.encodingTable;
            decodingTable = temp.decodingTable;
            ois.close();
            fis.close();
        } catch (Exception e) {
            System.out.println("Error:" + e);
        }
    }
}
