

module "backorder_model_registry"{
    source = "./backorder_model_bucket"    
}


module "backorder_ec2_instance"{
    source = "./backorder_ec2"    
}