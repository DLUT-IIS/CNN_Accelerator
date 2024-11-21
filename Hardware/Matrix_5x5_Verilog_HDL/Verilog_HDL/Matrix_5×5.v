//Matrix_5x5模块，stride=1，padding=2或padding=0；
module Matrix_5x5#(
		parameter		Data_width	=	8	 //数据位宽
)(
input   						i_clk										,
input   						i_rst_n										,
input							i_stride									,	//i_stride=0,步长为1;（此处仅用到stride=1的情况，没有stride=2的情况）
input							i_padding									,	//i_paddding=0,加两圈0；i_padding=1,不加padding
input		[7:0]				i_hs_num									,
input		[7:0]				i_vs_num									,
input 							i_data_en									,
input   	[Data_width-1:0]	i_data_in									,
output 	reg						o_read_en									,
output	reg	[10:0]				o_data_addr									,
output	reg						o_Mat5x5_data_en							,	
output  	[Data_width-1:0]	o_c0_0,o_c0_1,o_c0_2,o_c0_3,o_c0_4			, 
output  	[Data_width-1:0]	o_c1_0,o_c1_1,o_c1_2,o_c1_3,o_c1_4			, 
output  	[Data_width-1:0]	o_c2_0,o_c2_1,o_c2_2,o_c2_3,o_c2_4			,
output  	[Data_width-1:0]	o_c3_0,o_c3_1,o_c3_2,o_c3_3,o_c3_4			, 
output  	[Data_width-1:0]	o_c4_0,o_c4_1,o_c4_2,o_c4_3,o_c4_4	
);
/************************************************ reg **********************************************************/	
reg   [Data_width-1:0]				r_c0_0,r_c0_1,r_c0_2,r_c0_3,r_c0_4;
reg   [Data_width-1:0]				r_c1_0,r_c1_1,r_c1_2,r_c1_3,r_c1_4;
reg   [Data_width-1:0]				r_c2_0,r_c2_1,r_c2_2,r_c2_3,r_c2_4;
reg   [Data_width-1:0]				r_c3_0,r_c3_1,r_c3_2,r_c3_3,r_c3_4;
reg   [Data_width-1:0]				r_c4_0,r_c4_1,r_c4_2,r_c4_3,r_c4_4;
reg		[1:0] 						state							  ;
reg									o_Mat5x5_data_en_r				  ;
reg									fifo_rd			[3:0]			  ;
reg		[3:0]						St								  ;
reg		[7:0]						hs_cnt							  ;//行计数器
reg 	[7:0]						vs_cnt							  ;//列数计数器	
reg		[7:0]						vs_cnt_d						  ;	
reg 								r_read_en						  ;
/************************************************ wire **********************************************************/	
wire						fifo_wr			[3:0]	;
wire	[Data_width-1:0]	fifo_datain		[3:0]	;
wire	[Data_width-1:0]	fifo_dataout	[3:0]	;
wire						fifo_full		[3:0]	;
wire						fifo_empty		[3:0]	;
wire	[7:0]				fifo_cnt		[3:0]	;
/************************************************ assign **********************************************************/	
//fifo的写控制信号
assign					o_c0_0=r_c0_0;
assign					o_c0_1=r_c0_1;
assign					o_c0_2=r_c0_2;
assign					o_c0_3=r_c0_3;
assign					o_c0_4=r_c0_4;
assign					o_c1_0=r_c1_0;
assign					o_c1_1=r_c1_1;
assign					o_c1_2=r_c1_2;
assign					o_c1_3=r_c1_3;
assign					o_c1_4=r_c1_4;
assign					o_c2_0=r_c2_0;
assign					o_c2_1=r_c2_1;
assign					o_c2_2=r_c2_2;
assign					o_c2_3=r_c2_3;
assign					o_c2_4=r_c2_4;
assign					o_c3_0=r_c3_0;
assign					o_c3_1=r_c3_1;
assign					o_c3_2=r_c3_2;
assign					o_c3_3=r_c3_3;
assign					o_c3_4=r_c3_4;
assign					o_c4_0=r_c4_0;
assign					o_c4_1=r_c4_1;
assign					o_c4_2=r_c4_2;
assign					o_c4_3=r_c4_3;
assign					o_c4_4=r_c4_4;
assign					fifo_wr[0]=o_read_en;
assign  				fifo_datain[0]=i_data_in;
assign					fifo_wr[1]=fifo_rd[0];
assign					fifo_datain[1]=fifo_dataout[0];
assign					fifo_wr[2]=fifo_rd[1];
assign					fifo_datain[2]=fifo_dataout[1];
assign					fifo_wr[3]=fifo_rd[2];
assign					fifo_datain[3]=fifo_dataout[2];
/************************************************ always **********************************************************/	
always@(posedge i_clk or negedge i_rst_n )
begin
	if(!i_rst_n)
		r_read_en<='b0;
	else
		r_read_en<=o_read_en;
end
always@(posedge i_clk or negedge i_rst_n )
begin
	if(!i_rst_n)
		vs_cnt_d<='b0;
	else
		vs_cnt_d<=vs_cnt;
end
always@(*)
begin
	if(i_padding=='d0)
		begin
				o_Mat5x5_data_en<=o_Mat5x5_data_en_r;
		end
	else if(i_padding=='d1)
		begin
				if(vs_cnt==3&&(St==4))
							o_Mat5x5_data_en='d0;
				else if(vs_cnt<='d2||hs_cnt==2||hs_cnt==3||hs_cnt==1||St==5||St==0||hs_cnt==4||vs_cnt_d>'d5)
					begin
						o_Mat5x5_data_en='d0;
					end
				else 
						o_Mat5x5_data_en=o_Mat5x5_data_en_r;
					
		end
end
//访问数据信息
always@(posedge i_clk or negedge i_rst_n)
begin
	if(!i_rst_n)
		begin
			o_read_en<=0;
			o_data_addr<=0;
		end	
	else if(((hs_cnt==i_hs_num-1)&&i_data_en)||St==4)	
		begin
			o_read_en<=0;
			o_data_addr<=o_data_addr;
		end
	else if(i_data_en)
		begin
			o_read_en<=1;
			o_data_addr<=o_data_addr+1;
			
		end
	else
		begin
			o_read_en<=0;
			o_data_addr<=o_data_addr;
		end
end
//buffer生成
genvar i;
generate
	for(i=0;i<=3;i=i+1) begin : row_fifo           
		sync_fifo u_fifo(
			.i_clk(i_clk),
			.i_rst_n(i_rst_n),
			.i_data(fifo_datain[i]),
			.i_w_en(fifo_wr[i]),
			.i_r_en(fifo_rd[i]),
			.o_fifo_cnt(fifo_cnt[i]),
			.o_buf_empty(fifo_empty[i]),
			.o_buf_full(fifo_full[i]),
			.o_data(fifo_dataout[i])
		);
	end
endgenerate
//fifo读数据控制
always@(posedge i_clk) 
begin
	if(!i_rst_n) begin
		hs_cnt<='h0;
		vs_cnt<='h0;
	
		
	end
//当一副图像全部读完后，行列计数器归零	
	else if(fifo_rd[0]&&(hs_cnt==i_hs_num)&&(vs_cnt==i_vs_num)) begin
		hs_cnt<='h0;
		vs_cnt<='h0;
		
	end
	else if(St==5)
	begin
		hs_cnt<='b0;
	end
//当读完一行像素时，列计数器加1，并行计数器归零
	else if(fifo_rd[0]&&(hs_cnt==i_hs_num-1)) begin
		hs_cnt<='h0;
		
		vs_cnt<=vs_cnt+'h1;

	end
//当读数据使能时，行计数器加1
	else if(fifo_rd[0]) begin
		
		hs_cnt<=hs_cnt+'h1;
		vs_cnt<=vs_cnt;
	end
//保持不变
	else begin
		
		hs_cnt<=hs_cnt;
		vs_cnt<=vs_cnt;
	end
end
//fifo写状态标定
always@(posedge i_clk) begin
    if(!i_rst_n)
        state<=4'd0;
    else if(vs_cnt<i_vs_num)
	    state<=4'd0;
	else if((vs_cnt==i_vs_num))//处理倒数第一行
		state<=4'd1;
	else
	    state<=4'd2;
end

//fifo读使能控制
//state=0:为去除最后一行都记为正常行
//state=1：为倒数第一行
always@(*)
begin
	case(state)
		4'd0:begin
			fifo_rd[0]=(fifo_cnt[0]>=i_hs_num)&&i_data_en&&o_read_en&&!fifo_full[1];
			fifo_rd[1]=(fifo_cnt[1]>=i_hs_num)&&i_data_en&&o_read_en&&!fifo_full[2];
			fifo_rd[2]=(fifo_cnt[2]>=i_hs_num)&&i_data_en&&o_read_en&&!fifo_full[3];
			fifo_rd[3]=(fifo_cnt[3]>=i_hs_num)&&i_data_en&&o_read_en;
		end
		4'd1:begin
			fifo_rd[0]=!fifo_empty[0]&&!fifo_full[1]&&o_read_en;
			fifo_rd[1]=!fifo_empty[1]&&!fifo_full[2]&&o_read_en;
			fifo_rd[2]=!fifo_empty[2]&&!fifo_full[3]&&(fifo_cnt[2]>=i_hs_num)&&o_read_en;
			fifo_rd[3]=!fifo_empty[3]&&(fifo_cnt[3]>=i_hs_num)&&o_read_en;
		end 
		4'd2:begin              
            fifo_rd[0]=1'b0;
			fifo_rd[1]=1'b0;
			fifo_rd[2]=1'b0;
			fifo_rd[3]=1'b0;
		end
		default:begin
			fifo_rd[0]=1'b0;
			fifo_rd[1]=1'b0;
			fifo_rd[2]=1'b0;
			fifo_rd[3]=1'b0;
		end
	endcase
end
//矩阵查找
always@(posedge i_clk)
begin
	if(!i_rst_n) begin
		St<='h0;
		o_Mat5x5_data_en_r<=1'b0;
		r_c0_0<='h0;	r_c0_1<='h0;	r_c0_2<='h0;	r_c0_3<='h0;	r_c0_4<='h0;	
		r_c1_0<='h0;	r_c1_1<='h0;	r_c1_2<='h0;	r_c1_3<='h0;	r_c1_4<='h0;																
		r_c2_0<='h0;	r_c2_1<='h0;	r_c2_2<='h0;	r_c2_3<='h0;	r_c2_4<='h0;
		r_c3_0<='h0;	r_c3_1<='h0;	r_c3_2<='h0;	r_c3_3<='h0;	r_c3_4<='h0;																
		r_c4_0<='h0;	r_c4_1<='h0;	r_c4_2<='h0;	r_c4_3<='h0;	r_c4_4<='h0;
	end
	case(St)
	//处理每一行第一个像素
		4'd0:begin
			o_Mat5x5_data_en_r<=1'b0;
			if(fifo_rd[1]&&fifo_rd[0]&&(hs_cnt=='h0))begin
				St<=4'd1;
				if(vs_cnt=='h0)begin//处理首行
					r_c0_0	<=8'd0;
					r_c1_0	<=8'd0;
					r_c2_0	<=8'd0;	
					r_c3_0	<=8'd0;
					r_c4_0	<=8'd0;
					
					r_c0_1	<=8'd0;   
					r_c1_1	<=8'd0;   
					r_c2_1	<=8'd0;					
					r_c3_1	<=8'd0;
					r_c4_1	<=8'd0; 
					
					r_c0_2	<=8'd0;   
					r_c1_2	<=8'd0;   
					r_c2_2	<=fifo_dataout[1];					
					r_c3_2	<=fifo_dataout[0];
					r_c4_2	<=i_data_in; 					
				end
				else if(vs_cnt=='h1)begin
					r_c0_0	<=8'd0;
					r_c1_0	<=8'd0;
					r_c2_0	<=8'd0;	
					r_c3_0	<=8'd0;
					r_c4_0	<=8'd0;
					
					r_c0_1	<=8'd0;   
					r_c1_1	<=8'd0;   
					r_c2_1	<=8'd0;					
					r_c3_1	<=8'd0;
					r_c4_1	<=8'd0; 
					
					r_c0_2	<=8'd0;   
					r_c1_2	<=fifo_dataout[2];   
					r_c2_2	<=fifo_dataout[1];					
					r_c3_2	<=fifo_dataout[0];
					r_c4_2	<=i_data_in;
				end
				else if((vs_cnt<i_vs_num-'d1)&&vs_cnt>'h1)begin
					r_c0_0	<=8'd0;
					r_c1_0	<=8'd0;
					r_c2_0	<=8'd0;	
					r_c3_0	<=8'd0;
					r_c4_0	<=8'd0;
					
					r_c0_1	<=8'd0;   
					r_c1_1	<=8'd0;   
					r_c2_1	<=8'd0;					
					r_c3_1	<=8'd0;
					r_c4_1	<=8'd0; 				
					r_c0_2	<=fifo_dataout[3];   
					r_c1_2	<=fifo_dataout[2];   
					r_c2_2	<=fifo_dataout[1];					
					r_c3_2	<=fifo_dataout[0];
					r_c4_2	<=i_data_in;			
				end
				else if(vs_cnt==i_vs_num-'d1)begin
					r_c0_0	<=8'd0;
					r_c1_0	<=8'd0;
					r_c2_0	<=8'd0;	
					r_c3_0	<=8'd0;
					r_c4_0	<=8'd0;
					
					r_c0_1	<=8'd0;   
					r_c1_1	<=8'd0;   
					r_c2_1	<=8'd0;					
					r_c3_1	<=8'd0;
					r_c4_1	<=8'd0; 
					
					r_c0_2	<=fifo_dataout[3];   
					r_c1_2	<=fifo_dataout[2];   
					r_c2_2	<=fifo_dataout[1];					
					r_c3_2	<=fifo_dataout[0];
					r_c4_2	<=8'd0;				
				
				end
				else if(vs_cnt==i_vs_num)begin
					r_c0_0	<=8'd0;
					r_c1_0	<=8'd0;
					r_c2_0	<=8'd0;	
					r_c3_0	<=8'd0;
					r_c4_0	<=8'd0;
					
					r_c0_1	<=8'd0;   
					r_c1_1	<=8'd0;   
					r_c2_1	<=8'd0;					
					r_c3_1	<=8'd0;
					r_c4_1	<=8'd0; 
					
					r_c0_2	<=fifo_dataout[3];   
					r_c1_2	<=fifo_dataout[2];   
					r_c2_2	<=fifo_dataout[1];					
					r_c3_2	<=8'd0;
					r_c4_2	<=8'd0;	
				
				
				end
				end
			
			end
	//处理每一行第二个像素		
		4'd1:begin
				if(fifo_rd[1]&&fifo_rd[0]&&(hs_cnt=='h1))begin
				St<=4'd2;
					if(vs_cnt=='h0)begin
					r_c0_3	<=8'd0;
					r_c1_3	<=8'd0;
					r_c2_3	<=fifo_dataout[1];	
					r_c3_3	<=fifo_dataout[0];
					r_c4_3	<=i_data_in;
					end
					else if(vs_cnt=='h1)begin
					r_c0_3	<=8'd0;
					r_c1_3	<=fifo_dataout[2];
					r_c2_3	<=fifo_dataout[1];	
					r_c3_3	<=fifo_dataout[0];
					r_c4_3	<=i_data_in;
					end
					else if((vs_cnt<i_vs_num-'d1)&&vs_cnt>'h1)begin
					r_c0_3	<=fifo_dataout[3];
					r_c1_3	<=fifo_dataout[2];
					r_c2_3	<=fifo_dataout[1];	
					r_c3_3	<=fifo_dataout[0];
					r_c4_3	<=i_data_in;
					end
					else if(vs_cnt==i_vs_num-'d1)begin
					r_c0_3	<=fifo_dataout[3];
					r_c1_3	<=fifo_dataout[2];
					r_c2_3	<=fifo_dataout[1];	
					r_c3_3	<=fifo_dataout[0];
					r_c4_3	<=8'd0;
					end
					else if(vs_cnt==i_vs_num)begin
					r_c0_3	<=fifo_dataout[3];
					r_c1_3	<=fifo_dataout[2];
					r_c2_3	<=fifo_dataout[1];	
					r_c3_3	<=8'd0;
					r_c4_3	<=8'd0;
					end
				end
			end
	//处理每一行第三个元素		
		4'd2:begin
				if(fifo_rd[1]&&fifo_rd[0]&&(hs_cnt=='h2))begin
					St<=4'd3;
				
					if(vs_cnt=='h0)begin
					r_c0_4	<=8'd0;
					r_c1_4	<=8'd0;
					r_c2_4	<=fifo_dataout[1];	
					r_c3_4	<=fifo_dataout[0];
					r_c4_4	<=i_data_in;	
					o_Mat5x5_data_en_r<=1'b1;					
					end
					else if(vs_cnt=='h1)begin
					r_c0_4	<=8'd0;
					r_c1_4	<=fifo_dataout[2];
					r_c2_4	<=fifo_dataout[1];	
					r_c3_4	<=fifo_dataout[0];
					r_c4_4	<=i_data_in;
					o_Mat5x5_data_en_r<=1'b1;
					end
					else if((vs_cnt<i_vs_num-'d1)&&vs_cnt>'h1)begin
					r_c0_4	<=fifo_dataout[3];
					r_c1_4	<=fifo_dataout[2];
					r_c2_4	<=fifo_dataout[1];	
					r_c3_4	<=fifo_dataout[0];
					r_c4_4	<=i_data_in;	
					o_Mat5x5_data_en_r<=1'b1;					
					end
					else if(vs_cnt==i_vs_num-'d1)begin
					r_c0_4	<=fifo_dataout[3];
					r_c1_4	<=fifo_dataout[2];
					r_c2_4	<=fifo_dataout[1];	
					r_c3_4	<=fifo_dataout[0];
					r_c4_4	<=8'd0;	
					o_Mat5x5_data_en_r<=1'b1;
					end
					else if(vs_cnt==i_vs_num)begin
					r_c0_4	<=fifo_dataout[3];
					r_c1_4	<=fifo_dataout[2];
					r_c2_4	<=fifo_dataout[1];	
					r_c3_4	<=8'd0;
					r_c4_4	<=8'd0;	
					o_Mat5x5_data_en_r<=1'b1;
					end
				end
																	
			end		
	//处理中间像素点
		4'd3:begin
				if(fifo_rd[1]&&fifo_rd[0]&&(hs_cnt==i_hs_num-'d1))
					St<=4'd4;
				else
					St<=St;
				if(fifo_rd[0])
					begin
					o_Mat5x5_data_en_r<=1'b1;
						r_c0_0<=r_c0_1;	r_c0_1<=r_c0_2;	r_c0_2<=r_c0_3;	r_c0_3<=r_c0_4;		
						r_c1_0<=r_c1_1;	r_c1_1<=r_c1_2;	r_c1_2<=r_c1_3;	r_c1_3<=r_c1_4;		
						r_c2_0<=r_c2_1;	r_c2_1<=r_c2_2;	r_c2_2<=r_c2_3;	r_c2_3<=r_c2_4;	
						r_c3_0<=r_c3_1;	r_c3_1<=r_c3_2;	r_c3_2<=r_c3_3;	r_c3_3<=r_c3_4;		
						r_c4_0<=r_c4_1;	r_c4_1<=r_c4_2;	r_c4_2<=r_c4_3;	r_c4_3<=r_c4_4;	
					if(vs_cnt=='h0)begin
						r_c0_4<='h0;
					    r_c1_4<='h0;
					    r_c2_4<=fifo_dataout[1];
					    r_c3_4<=fifo_dataout[0];
					    r_c4_4<=i_data_in;
					
					end
					else if(vs_cnt=='h1)begin
						r_c0_4<='h0;
					    r_c1_4<=fifo_dataout[2];
					    r_c2_4<=fifo_dataout[1];
					    r_c3_4<=fifo_dataout[0];
					    r_c4_4<=i_data_in;									
					end
					else if((vs_cnt<i_vs_num-'d1)&&vs_cnt>'h1)begin
						r_c0_4<=fifo_dataout[3];
					    r_c1_4<=fifo_dataout[2];
					    r_c2_4<=fifo_dataout[1];
					    r_c3_4<=fifo_dataout[0];
					    r_c4_4<=i_data_in;
					end
					else if(vs_cnt==i_vs_num-'d1)begin
						r_c0_4<=fifo_dataout[3];
					    r_c1_4<=fifo_dataout[2];
					    r_c2_4<=fifo_dataout[1];
					    r_c3_4<=fifo_dataout[0];
					    r_c4_4<=8'd0;
					end
					else if(vs_cnt==i_vs_num)begin
						r_c0_4<=fifo_dataout[3];
					    r_c1_4<=fifo_dataout[2];
					    r_c2_4<=fifo_dataout[1];
					    r_c3_4<=8'd0;
					    r_c4_4<=8'd0;
					end
				end
					
				 
		end
	//处理倒数第二个元素	
		4'd4:begin
				
				St<=4'd5;
					r_c0_0<=r_c0_1;	r_c0_1<=r_c0_2;	r_c0_2<=r_c0_3;	r_c0_3<=r_c0_4;r_c0_4<='h0;	
					r_c1_0<=r_c1_1;	r_c1_1<=r_c1_2;	r_c1_2<=r_c1_3;	r_c1_3<=r_c1_4;r_c1_4<='h0;	
					r_c2_0<=r_c2_1;	r_c2_1<=r_c2_2;	r_c2_2<=r_c2_3;	r_c2_3<=r_c2_4;r_c2_4<='h0;
					r_c3_0<=r_c3_1;	r_c3_1<=r_c3_2;	r_c3_2<=r_c3_3;	r_c3_3<=r_c3_4;r_c3_4<='h0;	
					r_c4_0<=r_c4_1;	r_c4_1<=r_c4_2;	r_c4_2<=r_c4_3;	r_c4_3<=r_c4_4;r_c4_4<='h0;				
		end
		4'd5:begin
			St<='d0;
			
				r_c0_0<=r_c0_1;	r_c0_1<=r_c0_2;	r_c0_2<=r_c0_3;	r_c0_3<=r_c0_4;r_c0_4<='h0;	
				r_c1_0<=r_c1_1;	r_c1_1<=r_c1_2;	r_c1_2<=r_c1_3;	r_c1_3<=r_c1_4;r_c1_4<='h0;	
				r_c2_0<=r_c2_1;	r_c2_1<=r_c2_2;	r_c2_2<=r_c2_3;	r_c2_3<=r_c2_4;r_c2_4<='h0;
				r_c3_0<=r_c3_1;	r_c3_1<=r_c3_2;	r_c3_2<=r_c3_3;	r_c3_3<=r_c3_4;r_c3_4<='h0;	
				r_c4_0<=r_c4_1;	r_c4_1<=r_c4_2;	r_c4_2<=r_c4_3;	r_c4_3<=r_c4_4;r_c4_4<='h0;	
		end
		default:begin
			St<='h0;
			o_Mat5x5_data_en_r<=1'b0;
			r_c0_0<='h0;	r_c0_1<='h0;	r_c0_2<='h0;	r_c0_3<='h0;	r_c0_4<='h0;	
			r_c1_0<='h0;	r_c1_1<='h0;	r_c1_2<='h0;	r_c1_3<='h0;	r_c1_4<='h0;																
			r_c2_0<='h0;	r_c2_1<='h0;	r_c2_2<='h0;	r_c2_3<='h0;	r_c2_4<='h0;
			r_c3_0<='h0;	r_c3_1<='h0;	r_c3_2<='h0;	r_c3_3<='h0;	r_c3_4<='h0;																
			r_c4_0<='h0;	r_c4_1<='h0;	r_c4_2<='h0;	r_c4_3<='h0;	r_c4_4<='h0;		
		end
	endcase
end
endmodule