package PingYingEncoder;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.FlowLayout;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

import java.io.*;

import PingYingEncoder.encoders.*;

class MainFrame extends JFrame implements ActionListener {

    JButton convertButton;
    JTextField input;
    JTextField output;
    JTextField pass;

    JRadioButton morse;
    JRadioButton encrypted;
    ButtonGroup encoderType;
    String selectedEncoder;

    JRadioButton encode;
    JRadioButton decode;
    ButtonGroup operation;
    String selectedOperation;

    JPanel northPanel;
    JPanel centerPanel;
    JPanel southPanel;

    // JButton saveEncoder;
    // JButton loadEncoder;

    MainFrame() {
        morse = new JRadioButton("Morse");
        morse.addActionListener(this);
        encrypted = new JRadioButton("Encrypted");
        encrypted.addActionListener(this);
        encoderType = new ButtonGroup();
        encoderType.add(morse);
        encoderType.add(encrypted);

        encode = new JRadioButton("Encode");
        encode.addActionListener(this);
        decode = new JRadioButton("Decode");
        decode.addActionListener(this);
        operation = new ButtonGroup();
        operation.add(encode);
        operation.add(decode);

        convertButton = new JButton();
        convertButton.setBounds(20, 20, 10, 10);
        convertButton.addActionListener(this);
        convertButton.setText("Convert");
        convertButton.setFocusable(false);
        convertButton.setEnabled(true);

        input = new JTextField();
        input.setPreferredSize(new Dimension(250, 40));

        output = new JTextField();
        // output.setEnabled(false);
        output.setPreferredSize(new Dimension(250, 40));

        pass = new JTextField();
        pass.setPreferredSize(new Dimension(250, 40));
        pass.setVisible(false);

        northPanel = new JPanel();
        northPanel.setLayout(new GridLayout(1, 3));
        northPanel.add(new JLabel("Select Operation"));
        northPanel.add(encode);
        northPanel.add(decode);

        centerPanel = new JPanel();
        centerPanel.setLayout(new FlowLayout());
        centerPanel.add(new JLabel("Select Encoder Type"));
        centerPanel.add(morse);
        centerPanel.add(encrypted);

        southPanel = new JPanel();
        southPanel.setLayout(new GridLayout(6, 1));
        southPanel.add(new JLabel("Input"));
        southPanel.add(input);
        southPanel.add(pass);
        southPanel.add(new JLabel("Output"));
        southPanel.add(output);
        southPanel.add(convertButton);

        this.setLayout(new BorderLayout());
        this.setTitle("Encoder");
        this.setSize(350, 350);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setResizable(true);
        this.getContentPane().setBackground(new Color(223, 150, 250));

        this.add(northPanel, BorderLayout.NORTH);
        this.add(centerPanel, BorderLayout.CENTER);
        this.add(southPanel, BorderLayout.SOUTH);
        this.setVisible(true);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == convertButton) {
            Encoder encoder;
            if (selectedEncoder == "morse") {
                encoder = new MorseEncoder();
            } else if (selectedEncoder == "encrypted") {
                encoder = new EncryptedEncoder(pass.getText());
            } else {
                encoder = new MorseEncoder();
            }
            if (selectedOperation == "encode") {
                output.setText(encoder.encode(input.getText()));

            } else if (selectedOperation == "decode") {
                output.setText(encoder.decode(input.getText()));
            }
        } else if (e.getSource() == morse) {
            System.out.println("in encode");
            selectedEncoder = "morse";
            pass.setVisible(false);
        } else if (e.getSource() == encrypted) {
            selectedEncoder = "encrypted";
            pass.setVisible(true);
        } else if (e.getSource() == encode) {
            selectedOperation = "encode";
        } else if (e.getSource() == decode) {
            selectedOperation = "decode";
        }
    }
}
