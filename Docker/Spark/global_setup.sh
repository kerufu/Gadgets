export TZ=US/Pacific
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

apt-get update
apt-get upgrade
apt-get -y -qq install openjdk-8-jre openjdk-8-jdk openjdk-8-dbg python3 python3-pip inetutils-ping sudo openssh-client openssh-server scala

pip3 install -r python_requirements.txt
mkdir /node/runtime
mkdir /node/runtime/hadoop
mkdir /node/runtime/hadoop/tmp /node/runtime/hadoop/hdfs_name /node/runtime/hadoop/hdfs_data

echo -e "172.16.0.1 node1 \n172.16.0.2 node2 \n172.16.0.3 node3 \n172.16.0.4 node4" >> /etc/hosts
mkdir ~/.ssh
mv ssh_keys/authorized_keys ~/.ssh/authorized_keys