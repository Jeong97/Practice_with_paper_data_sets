clear all;
 path ='D:\����\ʦ�ֲ���\�������\Cell2\cycle1600\';
%  cd(path);
file =dir(strcat(path,'*'));
M= size(file,1);
for i=3:M    
    filename1=file(i).name;%ȡ����һ���ļ�������         
    disp(filename1);
    path2=strcat(path,filename1,'\');
    DIR=dir(strcat(path2,'*.txt'));
    len=length(DIR);
    cd(path2);
    for j=1:len
     old_name=DIR(j).name;
    new_name=strrep(old_name,'500','1600');
    system(['rename ' old_name ' ' new_name]);
    end
%     movefile(old_name,new_name)
end  