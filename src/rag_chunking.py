from langchain_text_splitters import RecursiveCharacterTextSplitter;

toy : str = ("Citi provides 17 weeks of paid parental leave.\n"
             "45 days of priviledge and sick leaves are "
             "credited to the employee at the beginning of the year.\n"
             "Meals for travel are reimbursed for $38 a day.\n");


splitter = RecursiveCharacterTextSplitter(chunk_size=25,
                                          chunk_overlap=0,
                                          ).split_text(toy);

print(splitter);

