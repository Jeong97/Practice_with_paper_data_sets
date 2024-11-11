clear all;
 path ='D:\桌面\师兄材料\电池数据\Cell2\';
%  cd(path);
fileDIR =dir(strcat(path,'*'));
w= size(fileDIR,1);
% W=(1:w)';
% % WWW=fileDIR(3).name;
% W1=num2str(W);
% W1(8)=fileDIR(5).name
% for i=6:w
%     W(i)=num2str(fileDIR(i).name);
% end
% % ooo=strfind(fileDIR.name,'cycle100');
oldcycle='975';
newcycle='3500';
for p=38
    filename=fileDIR(p).name;  %第p个子文件夹的文件夹名
     path1=strcat(path,filename,'\');
    file=dir(strcat(path1,'*')); %第一层文件夹1-50
     M=length(file);
for i=3:M    
    filename1=file(i).name;%取出第一个文件的名称         
    disp(filename1);
    path2=strcat(path1,filename1,'\');
    DIR=dir(strcat(path2,'*.txt'));
    len=length(DIR);
    cd(path2);
    for j=1:len
     old_name=DIR(j).name;
    new_name=strrep(old_name,oldcycle,newcycle);
    system(['rename ' old_name ' ' new_name]);
    end
%     movefile(old_name,new_name)
end  
end
cd(path);