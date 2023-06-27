#########################################################
    if args.require_image:
        test_ims = test_X[1].astype(np.float32)          # test_ims 手部图片数据 
        test_X = test_X[0]



    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)   # b*36*t->b*t*36
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)    #b*512*t->b*t*512



    #对测试集数据进行标准化处理
    body_mean_X = preprocess['body_mean_X']      #body 均值
    body_std_X = preprocess['body_std_X']        #body 方差
    body_mean_Y = preprocess['body_mean_Y']     #hand 均值
    body_std_Y = preprocess['body_std_Y']       #hand方差

    test_X = (test_X - body_mean_X) / body_std_X   #对body数据进行标准化
    test_Y = (test_Y - body_mean_Y) / body_std_Y    #对hand数据进行标准化
    ## DONE load/prepare data from external files


    ## pass loaded data into training
    # inputData = Variable(torch.from_numpy(test_X)).cuda()       #body数据 处理成可导
    # outputGT = Variable(torch.from_numpy(test_Y)).cuda()        #hand数据  真值
    imsData = None
    if args.require_image:
        imsData = Variable(torch.from_numpy(test_ims)).cuda()  #手部resnet处理






        

    # standardize           #载入模型预处理的核心数据
    checkpoint_dir = os.path.split(pretrain_model)[0]     #预训练所在目录
    model_tag = os.path.basename(args.checkpoint).split(args.pipeline)[0]        #从命令行参数中提取出模型标记 去掉arm2wh   
    #模型目录 模型tag arm2wh_preprocess_core.npz  npz文件在训练时已经生成 训练集的数据
    preprocess = np.load(os.path.join(checkpoint_dir,'{}{}_preprocess_core.npz'.format(model_tag, args.pipeline)))








   # output = model(inputData, image_=imsData) #####
    ################################################
    testLoss = 0
    total_mpjre = 0.  
    total_mpjpe = 0.

    batchinds = np.arange(test_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)     #批次数量

    for bii, bi in enumerate(batchinds):   
        ## setting batch data
        idxStart = bi * args.batch_size     #当前样本序号
        inputData_np = test_X[idxStart:(idxStart + args.batch_size), :, :] #取出body验证集中 当前批次的所有样本   
        outputData_np = test_Y[idxStart:(idxStart + args.batch_size), :, :]  #取出hand验证集中 当前批次的所有样本  真实值
        inputData = Variable(torch.from_numpy(inputData_np)).to(args.device)   #转换为pytorch张量
        outputGT = Variable(torch.from_numpy(outputData_np)).to(args.device)

        if args.require_image:
            imsData_np =imsData[idxStart:(idxStart + args.batch_size), :, :]  #取出手部resnet验证集中 当前批次的所有样本
            imsData = Variable(torch.from_numpy(imsData_np)).to(args.device)

        ## DONE setting batch data
        with torch.no_grad():    #关闭梯度计算
            total = torch.tensor([]).to(args.device)
            for i in range(inputData.shape[1]):   #seq_length次
                output =model(seq_input=inputData, hand_input=total, img_input=imsData, inference = True)  #前向传播
                # print(output.shape)
                output = generator.out(output[:,-1,:])
                output = einops.rearrange(output, 'b (t c) -> b t c', t = 1)  # 预测结果 b*1*(seq*output_dim)
                # print(output.shape)
                total = torch.concat([total,output],dim = 1)    #total 就包含了之前累积的输出和当前 output 的输出，形成了更长的序列。
                # print(total.shape)

            output = total.contiguous().view(-1, total.size(-1))  #(batchsize*seqlength)*output_dim
            # output = generator(seq_input=inputData, hand_input=outputGT,img_input=imsData)
            # output = generator.out(output)
            # output = output.contiguous().view(-1, output.size(-1))
            outputGT = outputGT.contiguous().view(-1, outputGT.size(-1))
            g_loss = criterion(output, outputGT)
            testLoss += g_loss.item()

            mpjre, mpjpe = evaluate_prediction(output, outputGT, 
                                                    reverse=args.reverse, alerady_std=args.alerady_std, is_adam=args.is_adam,
                                                    mean=torch.from_numpy(body_mean_Y).to(args.device), std=torch.from_numpy(body_std_Y).to(args.device),device=args.device)

        total_mpjre += mpjre 
        total_mpjpe += mpjpe

    testLoss /= totalSteps
    mpjre_mean = total_mpjre / totalSteps
    mpjpe_mean = total_mpjpe / totalSteps


    # error = criterion(output, outputGT).data
    

    
    ## DONE pass loaded data into training


    ## preparing output for saving
    # output_np = output.data.cpu().numpy()
    # output_gt = outputGT.data.cpu().numpy()
    # output_np = output_np * body_std_Y + body_mean_Y
    # output_gt = output_gt * body_std_Y + body_mean_Y
    # output_np = np.swapaxes(output_np, 1, 2).astype(np.float32)
    # output_gt = np.swapaxes(output_gt, 1, 2).astype(np.float32)
    # mpjre, mpjpe = evaluate_prediction(output, outputGT, 
    #                                             reverse=args.reverse, alerady_std=args.alerady_std, is_adam=args.is_adam,
    #                                            mean=torch.from_numpy(body_mean_Y).to(args.device), std=torch.from_numpy(body_std_Y).to(args.device),device=args.device)
    print(">>> TOTAL ERROR: \n", testLoss)
    print(">>>MPJRE:  ,MPJPE:   ",mpjre_mean,mpjpe_mean)
    print('----------------------------------')
    #save_results(test_Y_paths, output_np, args.pipeline, args.base_path, tag=args.tag+str(error))
    ## DONE preparing output for saving


