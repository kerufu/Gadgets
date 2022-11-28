package PingYingEncoder.encoders;

import java.util.HashMap;

public class MorseEncoder extends Encoder {

    public MorseEncoder() {
        super();
        HashMap<Character, String> morse = new HashMap<Character, String>();
        morse.put('a', ".-");
        morse.put('b', "-...");
        morse.put('c', "-.-.");
        morse.put('d', "-..");
        morse.put('e', ".");
        morse.put('f', "..-.");
        morse.put('g', "--.");
        morse.put('h', "....");
        morse.put('i', "..");
        morse.put('j', ".---");
        morse.put('k', "-.-");
        morse.put('l', ".-..");
        morse.put('m', "--");
        morse.put('n', "-.");
        morse.put('o', "---");
        morse.put('p', ".--.");
        morse.put('q', "--.-");
        morse.put('r', ".-.");
        morse.put('s', "...");
        morse.put('t', "-");
        morse.put('u', "..-");
        morse.put('v', "...-");
        morse.put('w', ".--");
        morse.put('x', "-..-");
        morse.put('y', "-.--");
        morse.put('z', "--..");

        morse.put('1', ".----");
        morse.put('2', "..---");
        morse.put('3', "...--");
        morse.put('4', "....-");
        morse.put('5', ".....");
        morse.put('6', "-....");
        morse.put('7', "--...");
        morse.put('8', "---..");
        morse.put('9', "----.");
        morse.put('0', "-----");

        morse.put(' ', ".-.-..");
        morse.put('.', ".-.-.-");
        morse.put(':', "---...");
        morse.put(',', "--..--");
        morse.put(';', "-.-.-.");
        morse.put('?', "..--..");
        morse.put('=', "-...-");
        morse.put('\'', ".----.");
        morse.put('/', "-..-.");
        morse.put('!', "-.-.--");
        morse.put('-', "-....-");
        morse.put('_', "..--.-");
        morse.put('"', ".-..-.");
        morse.put('(', "-.--.");
        morse.put(')', "-.--.-");
        morse.put('$', "...-..-");
        morse.put('&', ".-...");
        morse.put('@', ".--.-.");
        morse.put('+', ".-.-.");

        this.setTables(morse);
    }

}
