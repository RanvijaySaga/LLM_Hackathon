
-- Drop table

-- DROP TABLE public.account_overview;

CREATE TABLE public.account_overview (
	account_number varchar(50) NOT NULL,
	transaction_date date NOT NULL,
	"indicator" varchar(1) NULL,
	amount numeric(10,2) NOT NULL,
	payment_method varchar(50) NULL,
	merchant varchar(50) NULL,
	description_lines varchar(500) NULL,
	expenset_type varchar(50) NULL,
	CONSTRAINT account_overview_pkey PRIMARY KEY (account_number)
);


-- Drop table

-- DROP TABLE public.expense_category;

CREATE TABLE public.expense_category (
	expense_type varchar(50) NOT NULL,
	expense_category varchar(1) NULL,
	CONSTRAINT expense_category_pkey PRIMARY KEY (expense_type)
);
