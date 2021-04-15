import gpizero
import sysfs_paths as sysfs
import subprocess
import time
import telnetlib as tel
import psutil

def getTelnetPower(SP2_tel, last_power):
    tel_dat = str(SP2_tel.read_very_eager())
    print("telnet reading:", tel_dat)
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2: findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power




DELAY = 0.2
out_fname = "problem3-1.txt"
header = "time V A W W/h temp"
header = "\t".join(header.split(' '))
out_file = open(out_fname, 'w')
out_file.write(header)
out_file.write('\n')
SP2_tel = tel.Telnet('192.168.4.1')
total_power = 0.0
#subprocess.Popen(command.split(' '))
for i in range(6000):
    start = time.time()
    
    total_power = getTelnetPower(SP2_tel, total_power)
    temp = gpizero.CPUTemperature().temperature
    
    fmt_str = '{}\t'*15
    out = fmt_str.format(time_stamp, total_power, str(temp))
    print(out)
    out_file.write(out)
    out_file.write('\n')
    elapsed = time.time()-start
    time.sleep(max(0, DELAY-elapsed))
