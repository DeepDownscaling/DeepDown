def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, G_iters=1, show_every=250, num_epochs=5):
    """
    Adapted from CS231n Homework 3's GAN notebook
    
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    # Move the models to the correct device (GPU if GPU is available)
    D = D.to(device=device)
    G = G.to(device=device)
    
    # Put models in training mode
    D.train()
    G.train()
    
    print("Expected num iters: ", len(loader_train)*num_epochs)
    G_content = np.zeros(len(loader_train)*num_epochs*G_iters+1)
    G_advers = np.zeros(len(loader_train)*num_epochs*G_iters+1)
    D_real_L = np.zeros(len(loader_train)*num_epochs+1)
    D_fake_L = np.zeros(len(loader_train)*num_epochs+1)
    
    
    tic = time()
    
    iter_count = 0
    G_iter_count = 0
    for epoch in range(num_epochs):
        for x,y in loader_train:
                
            high_res_imgs = y.to(device=device, dtype=dtype)
            logits_real = D(high_res_imgs)

            x.requires_grad_()
            low_res_imgs = x.to(device=device, dtype=dtype)
            fake_images = G(low_res_imgs)
            logits_fake = D(fake_images)
    
            # Update for the discriminator
            d_total_error, D_real_L[iter_count], D_fake_L[iter_count] = discriminator_loss(logits_real, logits_fake)
            D_solver.zero_grad()
            d_total_error.backward()
            D_solver.step()

            
            for i in range(G_iters):
                # Update for the generator
                fake_images = G(low_res_imgs)
                logits_fake = D(fake_images)
                gen_logits_fake = D(fake_images)
                weight_param = 1e-1 # Weighting put on adversarial loss
                g_error, G_content[G_iter_count], G_advers[G_iter_count] = generator_loss(fake_images, high_res_imgs, gen_logits_fake, weight_param=weight_param)
                G_solver.zero_grad()
                g_error.backward()
                G_solver.step()
                G_iter_count += 1
                
                # Experimenting
#                 g_error, g_content, g_advers = generator_loss(fake_images, high_res_imgs, gen_logits_fake, weight_param=1e3)
#                 G_content[G_iter_count] = g_content
#                 G_advers[G_iter_count] = g_advers
#                 G_solver.zero_grad()
#                 g_advers.backward()
#                 G_solver.step()

            ################## LOGGING-BEGIN #########################
            writer.add_scalar('D Loss (Real)', D_real_L[iter_count], iter_count)
            writer.add_scalar('D Loss (Fake)', D_fake_L[iter_count], iter_count)
            writer.add_scalar('D Loss (Total)', d_total_error, iter_count)
            writer.add_scalar('G Loss (Content)', G_content[iter_count], iter_count)
            writer.add_scalar('G Loss (Adv)', G_advers[iter_count], iter_count)
            writer.add_scalar('G Loss (Total)', g_error, iter_count)
            if iter_count % save_img_every == 0:
                # precip
                input_precip_grid = vutils.make_grid(x[0, 0, :, :])
                writer.add_image('Input precipitation', input_precip_grid, iter_count)
                output_precip_grid = vutils.make_grid(fake_images[0, 0, :, :])
                writer.add_image('Output precipitation', output_precip_grid, iter_count)
                true_precip_grid = vutils.make_grid(y[0, 0, :, :])
                writer.add_image('True precipitation', true_precip_grid, iter_count)
                # temp
                input_temp_grid = vutils.make_grid(x[0, 1, :, :])
                writer.add_image('Input temperature', input_temp_grid, iter_count)
                output_temp_grid = vutils.make_grid(fake_images[0, 1, :, :])
                writer.add_image('Output temperature', output_temp_grid, iter_count)
                true_temp_grid = vutils.make_grid(y[0, 1, :, :])
                writer.add_image('True temperature', true_temp_grid, iter_count)
            ################## LOGGING-END ###########################

            if (iter_count % show_every == 0):
                toc = time()
                print('Epoch: {}, Iter: {}, D: {:.4}, G: {:.4}, Time since last print (min): {:.4}'.format(epoch,iter_count,d_total_error.item(),g_error.item(), (toc-tic)/60 ))
                tic = time()
                plot_epoch(x, fake_images, y)
                plot_loss(G_content, G_advers, D_real_L, D_fake_L, weight_param)
                print()
            iter_count += 1
        torch.save(D.cpu().state_dict(), 'GAN_Discriminator_checkpoint_adversWP_1e-1.pt')
        torch.save(G.cpu().state_dict(), 'GAN_Generator_checkpoint_adversWP_1e-1.pt')
        # Move the models back onto the GPU?
        # Move the models to the correct device (GPU if GPU is available)
        D = D.to(device=device)
        G = G.to(device=device)
        # Put models in training mode
        D.train()
        G.train()
#     print("FINAL ITER COUNT: ", iter_count)