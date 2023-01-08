

module "backorder_model_registry"{
    source = "./backorder_model_bucket"    
}

module "backorder_artifact_bucket"{
    source = "./backorder_artifact_bucket"    
}

module "backorder_prediction_bucket"{
    source = "./backorder_prediction_bucket"    
}


module "backorder_ec2_instance"{
    source = "./backorder_ec2"    
}

module "backorder_ecr"{
    source = "./backorder_ecr"
}