package PingYingEncoder.encoders;

import java.util.HashMap;
import java.security.MessageDigest;

public class EncryptedEncoder extends Encoder {
    static transient MessageDigest md;
    transient HashMap<Character, String> table;
    transient String pass;
    static final transient char[] hexArray = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F' };

    public EncryptedEncoder(String p) {
        super();
        pass = p;
        try {
            md = MessageDigest.getInstance("MD5");
            table = new HashMap<Character, String>();
            addHash('a');
            addHash('b');
            addHash('c');
            addHash('d');
            addHash('e');
            addHash('f');
            addHash('g');
            addHash('h');
            addHash('i');
            addHash('j');
            addHash('k');
            addHash('l');
            addHash('m');
            addHash('n');
            addHash('o');
            addHash('p');
            addHash('q');
            addHash('r');
            addHash('s');
            addHash('t');
            addHash('u');
            addHash('v');
            addHash('w');
            addHash('x');
            addHash('y');
            addHash('z');

            addHash('1');
            addHash('2');
            addHash('3');
            addHash('4');
            addHash('5');
            addHash('6');
            addHash('7');
            addHash('8');
            addHash('9');
            addHash('0');

            addHash(' ');
            addHash('.');
            addHash(':');
            addHash(',');
            addHash(';');
            addHash('?');
            addHash('=');
            addHash('\'');
            addHash('/');
            addHash('!');
            addHash('-');
            addHash('_');
            addHash('"');
            addHash('(');
            addHash(')');
            addHash('$');
            addHash('&');
            addHash('@');
            addHash('+');

            this.setTables(table);
        } catch (Exception e) {
            System.out.println(e);
        }

    }

    private void addHash(char c) {
        try {
            byte[] bytesOfMessage = (pass + c).getBytes("UTF-8");
            byte[] theMD5digest = md.digest(bytesOfMessage);
            char[] hexChars = new char[theMD5digest.length * 2];
            int v;
            for (int j = 0; j < theMD5digest.length; j++) {
                v = theMD5digest[j] & 0xFF;
                hexChars[j * 2] = hexArray[v / 16];
                hexChars[j * 2 + 1] = hexArray[v % 16];
            }
            String hashString = new String(hexChars);
            table.put(c, hashString.substring(0, 4));
        } catch (Exception e) {
            System.out.println(e);
        }
    }

}
