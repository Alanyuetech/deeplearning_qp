import datetime
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import copy
import numpy as np

    
def f(args):
    # (local_st,loc_s,num,plate,conn,t0)=args   #参数打包为args
    (local_st,loc_s,num,plate,t0)=args   #参数打包为args
# def f(local_st,loc_s,num,plate,conn,t0):


    t = time.time() - t0
    res=[]
    # conn.send('local_st,loc_s: %s: start@%.2fs' % ((local_st,loc_s), t))




    local_now=[local_st[0]+loc_s[0],local_st[1]+loc_s[1]] 
    if (local_now[0]>=len(plate))|(local_now[0]<0)|(local_now[1]>=len(plate[0]))|(local_now[1]<0):  #判断没有超出边界
        return 
    elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
        return

    else:
        local_last=copy.deepcopy(local_st)   #第一步结束
        local_v=[local_now[0]-local_last[0],local_now[1]-local_last[1]]   #now-last 组成的向量
        if (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
            res.append(list(local_now))

            t = time.time() - t0
            # conn.send('local_st,loc_s: %s: finish@%.2fs, res = %s' %((local_st,loc_s), t, res))            

            return res
        else:
            pass


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
                return
            elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
                return            
            elif (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                res.append(list(local_now))

                t = time.time() - t0
                # conn.send('local_st,loc_s: %s: finish@%.2fs, res = %s' %((local_st,loc_s), t, res))                      
                return res
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
                return
            elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
                return
            elif (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                res.append(list(local_now))

                t = time.time() - t0
                # conn.send('local_st,loc_s: %s: finish@%.2fs, res = %s' %((local_st,loc_s), t, res))                   
                return res
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
                return
            elif (plate[local_now[0]][local_now[1]]==".")&(((local_now[0]==0)|(local_now[0]==len(plate)-1))&((local_now[1]==0)|(local_now[1]==len(plate[0])-1))):   #判断在不在四个角
                return
            elif (plate[local_now[0]][local_now[1]]==".")&(((local_v[0]==-1)&(local_now[0]==0))|((local_v[0]==1)&(local_now[0]==len(plate)-1))|((local_v[1]==-1)&(local_now[1]==0))|((local_v[1]==1)&(local_now[1]==len(plate[0])-1))): #在边上
                res.append(list(local_now))
                t = time.time() - t0
                # conn.send('local_st,loc_s: %s: finish@%.2fs, res = %s' %((local_st,loc_s), t, res))                   
                return res
            else:
                pass 
        elif plate[local_now[0]][local_now[1]]=="O":    
            return


def main():
    num =  76
    plate = ["E......O..","E.........","W..E...EW.","EE.OE.WWWO","O.WEOEWWW.",".OW..W....","W...EW.WE.",".E.W...OW.","OW...W.EEO","......W.W."]
    plate0 = ["E......O..","E.........","W..E...EW.","EE.OE.WWWO","O.WEOEWWW.",".OW..W....","W...EW.WE.",".E.W...OW.","OW...W.EEO","......W.W."]
    plate1 = ["E......O..","E.........","W..E...EW.","EE.OE.WWWO","O.WEOEWWW.",".OW..W....","W...EW.WE.",".E.W...OW.","OW...W.EEO","......W.W."]
    plate2 = ["E......O..","E.........","W..E...EW.","EE.OE.WWWO","O.WEOEWWW.",".OW..W....","W...EW.WE.",".E.W...OW.","OW...W.EEO","......W.W."]
    plate3 = ["E......O..","E.........","W..E...EW.","EE.OE.WWWO","O.WEOEWWW.",".OW..W....","W...EW.WE.",".E.W...OW.","OW...W.EEO","......W.W."]
        
    
    
    
    plate=list(map(list,plate))
    local=[]   

    t0 = time.time()
    num_cores = multiprocessing.cpu_count()


    for i in range(len(plate)):   #找到终点位置
        for j in range(len(plate[i])):
            if plate[i][j]=="O":
                local=local+[[i,j]]
            else:
                pass
    t = time.time() - t0
    print('local ready: %.2fs ' % ( t))    



    ############ multiprocessing.Pool()，map 方法##################################

    # p = multiprocessing.Pool(processes=1)
    # p_conn, c_conn = multiprocessing.Pipe()
    # params = []
    # for local_st in local[:]:  #循环每个终点位置
    #     for loc_s in [[0,-1],[0,1],[1,0],[-1,0]]:   #反向推导，每个终点位置最多输出4个第一步
    #         params.append((local_st,loc_s,num,plate,c_conn,t0))
    # res = p.map(f, params)    
    # p.close()
    # p.join()
    # print('output:')
    # while p_conn.poll():
    #     print(p_conn.recv())


    ############ multiprocessing.Pool()，apply_async 方法##################################

    # p = multiprocessing.Pool(processes=1)
    # p_conn, c_conn = multiprocessing.Pipe()
    # res = []
    # for local_st in local[:]:  #循环每个终点位置
    #     for loc_s in [[0,-1],[0,1],[1,0],[-1,0]]:   #反向推导，每个终点位置最多输出4个第一步           
    #         res.append(p.apply_async(f,(local_st,loc_s,num,plate,c_conn,t0)))
    # p.close()
    # p.join()
    # print('output:')
    # while p_conn.poll():
    #     print(p_conn.recv())
    # for i in range(len(res)):
    #     res[i] = res[i].get()

    ############ multiprocessing.Pool()，map_async 方法##################################

    # p = multiprocessing.Pool(processes=1)
    # p_conn, c_conn = multiprocessing.Pipe()
    # params = []
    # for local_st in local[:]:  #循环每个终点位置
    #     for loc_s in [[0,-1],[0,1],[1,0],[-1,0]]:   #反向推导，每个终点位置最多输出4个第一步
    #         params.append((local_st,loc_s,num,plate,c_conn,t0))
    # res = p.map_async(f, params).get()
    # p.close()
    # p.join()
    # print('output:')
    # while p_conn.poll():
    #     print(p_conn.recv())

    ############ multiprocessing.Pool()，imap 方法##################################
    
    # p = multiprocessing.Pool(processes=1)
    # p_conn, c_conn = multiprocessing.Pipe()
    # params = []
    # for local_st in local[:]:  #循环每个终点位置
    #     for loc_s in [[0,-1],[0,1],[1,0],[-1,0]]:   #反向推导，每个终点位置最多输出4个第一步
    #         params.append((local_st,loc_s,num,plate,c_conn,t0))
    # res = p.imap(f, params)
    # p.close()
    # p.join()
    # print('output:')
    # while p_conn.poll():
    #     print(p_conn.recv())

    ############ concurrent.futures.ProcessPoolExecutor 方法##################################
    params = []
    # for local_st in local[:]:
    #     for loc_s in [[0, -1], [0, 1], [1, 0], [-1, 0]]:
    #         params.append((local_st, loc_s, num, plate, t0))

    for local_st in local[:]:
        for loc_s in [[0, -1], [0, 1], [1, 0], [-1, 0]]:
            for ii in range(4):
                params.append((local_st, loc_s, num, locals()['plate'+str(ii)] , t0))    

    with ProcessPoolExecutor(max_workers=4) as executor:
        res = list(executor.map(f, params))


    # #删除res中的None
    res = [x for x in res if x is not None]
    res = [x[0] for x in res ]
    t = time.time() - t0
    print('final: %.2fs  res: %s' % ( t, res))
    


if __name__=='__main__':
    main()