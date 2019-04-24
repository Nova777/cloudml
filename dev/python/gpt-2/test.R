#install.packages("cloudml")

#library(cloudml)

#gcloud_install()

#gcloud_init()

# complex_model_m_gpu

job <- cloudml_train_python("c://projects/cloudml/dev/python/gpt-2/trainer/",
                            master_type = "complex_model_m_gpu")
