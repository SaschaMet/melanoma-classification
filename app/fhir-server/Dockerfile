FROM node:lts-alpine

# Update any outdated packages
RUN apk update

WORKDIR /app

# Move this separately so install get's cached
COPY package.json /app/package.json
RUN npm install --production

# Copy the remaining code over
COPY . /app/

EXPOSE 80
EXPOSE 443

# By default, use the production command
CMD npm start