import datetime
import time
s_time = time.time()
num =  76
plate = ["E......O..","E.........","W..E...EW.","EE.OE.WWWO","O.WEOEWWW.",".OW..W....","W...EW.WE.",".E.W...OW.","OW...W.EEO","......W.W."]
import copy
import numpy as np
plate=list(map(list,plate))
local=[]   
res=[]
for i in range(len(plate)):   #找到终点位置
    for j in range(len(plate[i])):
        if plate[i][j]=="O":
            local=local+[[i,j]]
        else:
            pass

for local_st in local[:]:  #循环每个终点位置
    for loc_s in [[0,-1],[0,1],[1,0],[-1,0]]:   #反向推导，每个终点位置最多输出4个第一步
        #打印当前时间
        print(datetime.datetime.now())  

        local_now=[local_st[0]+loc_s[0],local_st[1]+loc_s[1]] 
        if (local_now[0]>=len(plate))|(local_now[0]<0)|(local_now[1]>=len(plate[0]))|(local_now[1]<0):  #判断没有超出边界
            continue
        elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
            continue

        else:
            local_last=copy.deepcopy(local_st)   #第一步结束
            local_v=[local_now[0]-local_last[0],local_now[1]-local_last[1]]   #now-last 组成的向量
            if (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                res.append(list(local_now))
                print(local_st,loc_s,local_now)
                continue

        
        # if num-1==0:
        # if ((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1)):   #判断在不在四个角
        #     continue
        # elif ((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1)): #在边上
        #     res.append(list(local_now))
        #     continue



        for i in range(1,num):  #看剩下的步骤
            # print(plate,local_now)
            if plate[local_now[0]][local_now[1]]==".":
                local_last=copy.deepcopy(local_now)
                local_now=[local_now[0]+local_v[0],local_now[1]+local_v[1]]  #线性行走不用更新向量
                # 判断是否是入口
                if (local_now[0]>=len(plate))|(local_now[0]<0)|(local_now[1]>=len(plate[0]))|(local_now[1]<0):  #判断没有超出边界:
                    break
                elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
                    break            
                elif (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                    res.append(list(local_now))
                    print(local_st,loc_s,local_now)
                    break
                else:
                    pass                            

            elif plate[local_now[0]][local_now[1]]=="W":  #原来为逆时针，从终点出发变成顺时针
                if local_v==[0,1]:
                    local_v=[1,0]
                elif local_v==[1,0]:
                    local_v=[0,-1]
                elif local_v==[0,-1]:
                    local_v=[-1,0]
                elif local_v==[-1,0]:
                    local_v=[0,1]

                local_last=copy.deepcopy(local_now)
                local_now=[local_now[0]+local_v[0],local_now[1]+local_v[1]]

                # 判断是否是入口
                if (local_now[0]>=len(plate))|(local_now[0]<0)|(local_now[1]>=len(plate[0]))|(local_now[1]<0):  #判断没有超出边界:
                    break
                elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
                    break
                elif (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                    res.append(list(local_now))
                    print(local_st,loc_s,local_now)
                    break
                else:
                    pass                              

            elif plate[local_now[0]][local_now[1]]=="E":

                if local_v==[0,1]:
                    local_v=[-1,0]
                elif local_v==[-1,0]:
                    local_v=[0,-1]
                elif local_v==[0,-1]:
                    local_v=[1,0]
                elif local_v==[1,0]:
                    local_v=[0,1]

                local_last=copy.deepcopy(local_now)
                local_now=[local_now[0]+local_v[0],local_now[1]+local_v[1]]

                # 判断是否是入口
                if (local_now[0]>=len(plate))|(local_now[0]<0)|(local_now[1]>=len(plate[0]))|(local_now[1]<0):  #判断没有超出边界:
                    break
                elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
                    break
                elif (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                    res.append(list(local_now))
                    print(local_st,loc_s,local_now)
                    break
                else:
                    pass 
            elif plate[local_now[0]][local_now[1]]=="O":    
                break
            
e_time=time.time()
print(e_time-s_time,res)