# -*-coding:utf8 -*-
#import tensorflow as tf
import subprocess



def run_shell_cmd(shellcmd, encoding='utf8'):
    res = subprocess.Popen(shellcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    results = []
    while True:
        line = res.stdout.readline().decode(encoding).strip()
        if line == '' and res.poll() is not None:
            break
        else:
            if line.find("ERROR") == -1:
                results.append(line)

    return results



if __name__=='__main__':
    #command = "hadoop fs -ls hdfs://ns1013/user/recsys/suggest/app.db/hierarchical/tfrecord/* | grep part | awk '{print $NF}' 2>/dev/null"
    command = "ls data/tfrecord/* | grep part | awk '{print $NF}' 2>/dev/null"
    input_files = run_shell_cmd(command)
    print(len(input_files))
    print(input_files[0:2])
    train_file = ','.join(input_files)
    print(train_file)
