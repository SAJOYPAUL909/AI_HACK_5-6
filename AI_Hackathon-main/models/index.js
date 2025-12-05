// models/index.js
import sequelize from "../config/db.js";
import User from "./User.js";

const models = {
  User,
};

export { sequelize };
export default models;
