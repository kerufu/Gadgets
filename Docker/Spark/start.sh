$HADOOP_HOME/bin/hadoop namenode -format
$HADOOP_HOME/sbin/start-all.sh
$SPARK_HOME/sbin/start-all.sh
$SPARK_HOME/bin/spark-shell --master yarn --deploy-mode client