# syntax=docker/dockerfile:1
FROM ubuntu
WORKDIR /node/source
COPY . .

RUN bash ./global_setup.sh

ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV JRE_HOME $JAVA_HOME/jre

ENV SCALA_HOME /usr/share/java

ENV HADOOP_HOME /node/source/packages/hadoop-3.3.1
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV YARN_HOME $HADOOP_HOME
ENV YARN_CONF_DIR $HADOOP_CONF_DIR
ENV HDFS_NAMENODE_USER "root"
ENV HDFS_DATANODE_USER "root"
ENV HDFS_SECONDARYNAMENODE_USER "root"
ENV YARN_RESOURCEMANAGER_USER "root"
ENV YARN_NODEMANAGER_USER "root"

ENV SPARK_HOME /node/source/packages/spark-3.2.0-bin-hadoop3.2
ENV PYSPARK_PYTHON /usr/bin/python3
# ENV PATH $PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV SPARK_MASTER_IP 172.16.0.1
ENV SPARK_LOCAL_DIRS $SPARK_HOME
ENV SPARK_LIBARY_PATH .:$JAVA_HOME/lib:$JAVA_HOME/jre/lib:$HADOOP_HOME/lib/native

CMD (service ssh start || ls) && tail -F anything