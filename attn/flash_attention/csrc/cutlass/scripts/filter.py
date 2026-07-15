import sys
import datetime
#log path
file_path=sys.argv[1]
#kernel counts
kernel_sip=sys.argv[2]
contents=[]
with open(file_path, 'r') as file:
    for line in file:
        if '1,CUTLASS,gemm,cutlass_simt_f16_sgemm' in line:
            contents.append(line)
# print(len(contents))
save_index = 0
perf_max = 0
filter_save_max=[]
inner_loop = 0
for i in range(len(contents)):
    inner_loop += 1
    perf = contents[i].split(',')[-1]
    if perf_max < float(perf):
        save_index = i
        perf_max = float(perf)
    if inner_loop == int(kernel_sip):
        # print(contents[save_index])
        filter_save_max.append(contents[save_index])
        perf_max = 0
        inner_loop =0
# print(len(filter_save_max))  
save_path='./'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_gemm.csv';     
with open(save_path, 'w') as file:
    file.write('Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,D,alpha,beta,split_k_mode,split_k_slices,batch_count,raster_order,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs\n')
    for i in filter_save_max:
        file.write(i)
            